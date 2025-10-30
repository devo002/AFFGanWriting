# utils/util.py

import os
import math
import struct
from typing import Iterable, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage import draw


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def pt_xyrs_2_xyxy(state: torch.Tensor) -> torch.Tensor:
    # state: [B, N, 5] -> [B, N, 5] with (x0,y0,x1,y1) replacing (x,y,r,s)
    x = state[:, :, 1:2]
    y = state[:, :, 2:3]
    r = state[:, :, 3:4]
    s = state[:, :, 4:5]

    x0 = -torch.sin(r) * s + x
    y0 = -torch.cos(r) * s + y
    x1 = torch.sin(r) * s + x
    y1 = torch.cos(r) * s + y

    return torch.cat([state[:, :, 0:1], x0, y0, x1, y1], dim=2)


def pt_xyxy_2_xyrs(state: torch.Tensor) -> torch.Tensor:
    # state: [N, >=5] where [x0, y0, x1, y1, ...] -> [mx, my, theta, d, ...]
    x0 = state[:, 0:1]
    y0 = state[:, 1:2]
    x1 = state[:, 2:3]
    y1 = state[:, 3:4]

    dx = x0 - x1
    dy = y0 - y1
    d = torch.sqrt(dx ** 2.0 + dy ** 2.0) / 2.0

    mx = (x0 + x1) / 2.0
    my = (y0 + y1) / 2.0
    theta = -torch.atan2(dx, -dy)

    return torch.cat([mx, my, theta, d, state[:, 4:5]], dim=1)


def primeFactors(n: Union[int, np.integer, torch.Tensor]) -> Iterable[int]:
    """
    Return the prime factorization of n as a list (with multiplicities),
    e.g. primeFactors(36) -> [2, 2, 3, 3].
    """
    # robust cast (handles numpy scalars and torch tensors via .item())
    if hasattr(n, "item"):
        try:
            n = int(n.item())
        except Exception:
            n = int(n)
    else:
        n = int(n)

    if n < 2:
        return []

    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        while n % f == 0:
            factors.append(f)
            n //= f
        f += 2
    if n > 1:
        factors.append(n)
    return factors


def uniquePrimeFactors(n: Union[int, np.integer, torch.Tensor]) -> Iterable[int]:
    return sorted(set(primeFactors(n)))


def getGroupSize(channels: int) -> int:
    if channels >= 32:
        goalSize = 8
    else:
        goalSize = 4
    if channels % goalSize == 0:
        return goalSize
    factors = primeFactors(channels)
    bestDist = 9999
    bestGroup = 1
    for f in factors:
        if abs(f - goalSize) <= bestDist:  # favor larger
            bestDist = abs(f - goalSize)
            bestGroup = f
    return int(bestGroup)


# ---------------------------------------------------------------------------
# Tensor/NumPy conversion helpers
# ---------------------------------------------------------------------------

def _to_tensor_nchw(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert input to a torch.float32 tensor in NCHW (CPU).
    Accepts:
      - HxW (np/tensor) -> 1x1xH xW
      - NxHxW -> Nx1xH xW
      - NxHxWxC -> NxCxH xW
      - NxCxH xW -> as-is
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x))
    elif torch.is_tensor(x):
        t = x
    else:
        raise TypeError("makeMask: image must be numpy array or torch.Tensor")

    if t.ndim == 2:  # H,W
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        # assume N,H,W or H,W,C
        if t.shape[-1] in (1, 3) and (t.shape[0] != 1 and t.shape[1] != 1):
            # H,W,C -> 1,C,H,W
            t = t.permute(2, 0, 1).unsqueeze(0)
        else:
            # N,H,W -> N,1,H,W
            t = t.unsqueeze(1)
    elif t.ndim == 4 and t.shape[1] not in (1, 3) and t.shape[-1] in (1, 3):
        # channels-last -> channels-first
        t = t.permute(0, 3, 1, 2)

    return t.contiguous().to(dtype=torch.float32, device="cpu")


# ---------------------------------------------------------------------------
# Mask creation for line images
# ---------------------------------------------------------------------------

def makeMask(image: Union[np.ndarray, torch.Tensor], post=list(), random: Union[bool, str] = False
             ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    """
    Builds a soft mask used by the line pipeline.
    Returns:
        blurred_mask (Tensor N1HW, float32),
        top_and_bottom (Tensor N2W or None),
        centersV_t (Tensor N1W or None)
    """
    if post is None:
        post = []
    if isinstance(post, str):
        # allow a single token
        post = [] if post.lower() == "none" else [post]

    # Always operate on a tensor
    image = _to_tensor_nchw(image)
    batch_size = image.size(0)

    # Morphology kernel setup
    if random:
        morph_kernel_dilate = 2 * np.random.randint(8, 20) + 1
        morph_kernel_errode = morph_kernel_dilate + (2 * np.random.randint(-3, 4) if random == 'more' else 0)
        h_kernel = 2 * np.random.randint(10, 20) + 1
        v_kernel = h_kernel // 4 if (h_kernel // 4) % 2 == 1 else h_kernel // 4 + 1
    else:
        morph_kernel_dilate = 25
        morph_kernel_errode = 25
        h_kernel = 31
        v_kernel = h_kernel // 4

    morph_diff = morph_kernel_errode - morph_kernel_dilate
    morph_padding_errode = max(0, morph_diff // 2)
    morph_padding_dilate = max(0, -morph_diff // 2)
    h_padding = h_kernel // 2
    v_padding = v_kernel // 2

    # Pool/blur ops
    if len(post) > 0 and post[0] == 'true':
        post = post[1:]
        h_kernel = v_kernel = 3
        h_padding = v_padding = 1
        pool = torch.nn.MaxPool2d((v_kernel, h_kernel), stride=1, padding=(v_padding, h_padding))
        blur_kernel = 3
        blur_padding = blur_kernel // 2
        blur = torch.nn.AvgPool2d((blur_kernel, blur_kernel), stride=1, padding=(blur_padding, blur_padding))
    else:
        pool = torch.nn.MaxPool2d((v_kernel, h_kernel), stride=1, padding=(v_padding, h_padding))
        blur_kernel = 31
        blur_padding = blur_kernel // 2
        blur = torch.nn.AvgPool2d((blur_kernel // 4, blur_kernel // 4),
                                  stride=1, padding=(blur_padding // 4, blur_padding // 4))

    # Initial pooled map (tensor)
    pt_img = pool(image)
    out = torch.empty_like(image)

    # Cumulative max trick (NumPy side), then back to tensor
    for i in range(batch_size):
        pt_img_b = pt_img[i, 0].detach().cpu().numpy()
        cummax_img0 = np.maximum.accumulate(pt_img_b, axis=0)
        cummax_img1 = np.maximum.accumulate(pt_img_b[::-1], axis=0)[::-1]
        cummax_img2 = np.maximum.accumulate(pt_img_b, axis=1)
        cummax_img3 = np.maximum.accumulate(pt_img_b[::-1], axis=1)[::-1]
        result = np.minimum(np.minimum(cummax_img0, cummax_img1),
                            np.minimum(cummax_img2, cummax_img3))
        out[i, 0] = torch.from_numpy(result).to(dtype=out.dtype)

    # Post operations
    for task in post:
        if task == 'thresh':
            out = (out > 0.1).to(out.dtype)
        elif task == 'smaller':
            morph_kernel_dilate = morph_kernel_dilate // 2 + 1
            morph_kernel_errode = morph_kernel_errode // 2 + 1
        elif task == 'errode':
            weights = torch.ones(1, 1, morph_kernel_errode, morph_kernel_errode, dtype=torch.float32)
            out = F.conv2d(out, weights, stride=1)
            out = (out >= (morph_kernel_errode ** 2)).to(out.dtype)
        elif task == 'errodeCircle':
            weights = torch.zeros(1, 1, morph_kernel_errode, morph_kernel_errode, dtype=torch.float32)
            r = morph_kernel_errode // 2
            for x in range(morph_kernel_errode):
                for y in range(morph_kernel_errode):
                    weights[0, 0, y, x] = float(((y - r) ** 2 + (x - r) ** 2) <= (r ** 2))
            out = F.conv2d(out, weights, stride=1, padding=morph_padding_errode)
            out = (out >= weights.sum()).to(out.dtype)
        elif task == 'dilate':
            weights = torch.ones(1, 1, morph_kernel_dilate, morph_kernel_dilate, dtype=torch.float32)
            out = F.conv_transpose2d(out, weights, stride=1)
            out = (out > 0.1).to(out.dtype)
        elif task == 'dilateCircle':
            weights = torch.zeros(1, 1, morph_kernel_dilate, morph_kernel_dilate, dtype=torch.float32)
            r = morph_kernel_dilate // 2
            for x in range(morph_kernel_dilate):
                for y in range(morph_kernel_dilate):
                    weights[0, 0, y, x] = float(((y - r) ** 2 + (x - r) ** 2) <= (r ** 2))
            out = F.conv_transpose2d(out, weights, stride=1, padding=morph_padding_dilate)
            out = (out > 0.1).to(out.dtype)
        elif task == 'distance':
            # compute centerline distance map in NumPy
            arr = out.detach().cpu().numpy()
            height, width = arr.shape[2], arr.shape[3]
            window = 3 * height
            dists = np.empty(arr.shape, np.float32)
            for b in range(batch_size):
                line_im = np.ones((height, width), np.uint8)

                medians = []
                sum_x = 0.0
                sum_y = 0.0
                count = 1.0
                y_idx = np.arange(height)[:, None].repeat(window, axis=1)
                x_idx = np.arange(window)[None, :].repeat(height, axis=0)

                for x_start in range(0, width - window, window // 2):
                    on = arr[b, 0, :, x_start:x_start + window].sum()
                    if on > 0:
                        med_x = (x_idx * arr[b, 0, :, x_start:x_start + window]).sum() / on + x_start
                        med_y = (y_idx * arr[b, 0, :, x_start:x_start + window]).sum() / on
                        medians.append((med_y, med_x))
                        sum_x += med_x
                        sum_y += med_y
                        count += 1.0

                med_x = sum_x / count
                med_y = sum_y / count

                if len(medians) > 1:
                    slope = (medians[1][0] - medians[0][0]) / (medians[1][1] - medians[0][1])
                    distance = -medians[0][1]
                    front_point = [(med_y + medians[0][0] + slope * distance) / 2, 0]
                    slope = (medians[-1][0] - medians[-2][0]) / (medians[-1][1] - medians[-2][1])
                    distance = width - 1 - medians[-1][1]
                    last_point = [(med_y + medians[-1][0] + slope * distance) / 2, width - 1]
                    if last_point[0] < 0 or last_point[0] >= height:
                        last_point = (med_y, width - 1)
                else:
                    front_point = [med_y, med_x]
                    last_point = [med_y, med_x]

                medians = [front_point] + medians + [last_point]
                # repair NaNs
                for i in range(0, len(medians) - 1):
                    if math.isnan(medians[i][0]):
                        medians[i][0] = medians[i + 1][0]
                    if math.isnan(medians[i][1]):
                        medians[i][1] = medians[i + 1][1]
                for i in range(len(medians) - 1, 0, -1):
                    if math.isnan(medians[i][0]):
                        medians[i][0] = medians[i - 1][0]
                    if math.isnan(medians[i][1]):
                        medians[i][1] = medians[i - 1][1]

                for i in range(1, len(medians)):
                    rr, cc = draw.line(int(medians[i - 1][0]), int(medians[i - 1][1]),
                                       int(medians[i][0]), int(medians[i][1]))
                    line_im[rr, cc] = 0

                dist = distance_transform_edt(line_im)
                dists[b] = dist

            max_dist = height // 2
            dists /= max_dist
            dists[dists > 1] = 1
            dists = 1 - dists
            out = torch.from_numpy(dists).to(dtype=out.dtype)

        else:
            raise NotImplementedError(f'unknown makeMask post operation: {task}')

    # Optional centerline stats if any post-op requested
    if len(post) > 0:
        centersV_t = torch.from_numpy(getCenterValue(out))  # [N,W]
        centerV = centersV_t[:, None, ...]                  # [N,1,W]

        height = out.size(2)
        width = out.size(3)

        ranges = (torch.arange(height) + 1)[None, None, ..., None].expand(out.size(0), -1, -1, width)
        mask_ranges = ranges * out.long()
        bottom = mask_ranges.argmax(dim=2)
        bottom_not_valid = 0 == mask_ranges.max(dim=2)[0]

        mask_ranges = ((height + 1) - ranges) * out.long()
        top = mask_ranges.argmax(dim=2)
        top_not_valid = 0 == mask_ranges.max(dim=2)[0]

        top_and_bottom = torch.cat((centerV - top.float(), bottom.float() - centerV), dim=1)
        top_and_bottom[:, 0][top_not_valid[:, 0, :]] = 0
        top_and_bottom[:, 1][bottom_not_valid[:, 0, :]] = 0

        out = 2 * out.float() - 1
    else:
        top_and_bottom = None
        centersV_t = None

    return blur(out), top_and_bottom, centersV_t


def getCenterValue(mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Compute per-column vertical center (y) of the mask (N1HW).
    Returns np.ndarray shape [N, W].
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    batch_size = mask.shape[0]
    height = mask.shape[2]
    width = mask.shape[3]
    window = 3 * height

    centers = np.zeros([batch_size, width], np.float32)
    centers[:] = height / 2

    for b in range(batch_size):
        medians = []
        sum_x = 0.0
        sum_y = 0.0
        count = 1.0

        y_indexes = np.arange(height)[:, None].repeat(window, axis=1)
        x_indexes = np.arange(window)[None, :].repeat(height, axis=0)

        for x_start in range(0, width - window, window // 2):
            on = mask[b, 0, :, x_start:x_start + window].sum()
            if on > 0:
                med_x = (x_indexes * mask[b, 0, :, x_start:x_start + window]).sum() / on + x_start
                med_y = (y_indexes * mask[b, 0, :, x_start:x_start + window]).sum() / on
                medians.append((med_y, med_x))
                sum_x += med_x
                sum_y += med_y
                count += 1.0

        if len(medians) > 1:
            med_x = sum_x / count
            med_y = sum_y / count
            slope = (medians[1][0] - medians[0][0]) / (medians[1][1] - medians[0][1])
            distance = -medians[0][1]
            front_point = [(med_y + medians[0][0] + slope * distance) / 2, 0]
            slope = (medians[-1][0] - medians[-2][0]) / (medians[-1][1] - medians[-2][1])
            distance = width - 1 - medians[-1][1]
            last_point = [(med_y + medians[-1][0] + slope * distance) / 2, width - 1]
            if last_point[0] < 0 or last_point[0] >= height:
                last_point = (med_y, width - 1)
        else:
            on = mask[b, 0].sum()
            if on == 0:
                front_point = [height / 2, 0]
                last_point = [height / 2, width - 1]
            else:
                y_indexes = np.arange(height)[:, None].repeat(width, axis=1)
                x_indexes = np.arange(width)[None, :].repeat(height, axis=0)
                med_x = (x_indexes * mask[b, 0]).sum() / on
                med_y = (y_indexes * mask[b, 0]).sum() / on
                front_point = [med_y, 0]
                last_point = [med_y, width - 1]

        medians = [front_point] + medians + [last_point]
        for i in range(0, len(medians) - 1):
            if math.isnan(medians[i][0]):
                medians[i][0] = medians[i + 1][0]
            if math.isnan(medians[i][1]):
                medians[i][1] = medians[i + 1][1]
        for i in range(len(medians) - 1, 0, -1):
            if math.isnan(medians[i][0]):
                medians[i][0] = medians[i - 1][0]
            if math.isnan(medians[i][1]):
                medians[i][1] = medians[i - 1][1]

        for i in range(1, len(medians)):
            rr, cc = draw.line(int(medians[i - 1][0]), int(medians[i - 1][1]),
                               int(medians[i][0]), int(medians[i][1]))
            centers[b][cc] = rr

    return centers


# ---------------------------------------------------------------------------
# Image size util (binary-safe for Python3)
# ---------------------------------------------------------------------------

class UnknownImageFormat(Exception):
    pass


def get_image_size(file_path: str) -> Tuple[int, int]:
    """
    Return (width, height) for a given image file without fully decoding.
    Supports GIF/PNG/JPEG. Binary-safe for Python3.
    """
    size = os.path.getsize(file_path)
    with open(file_path, "rb") as input_f:
        height = -1
        width = -1
        data = input_f.read(25)

        if (size >= 10) and data[:6] in (b"GIF87a", b"GIF89a"):
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith(b"\211PNG\r\n\032\n")
              and (data[12:16] == b"IHDR")):
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith(b"\211PNG\r\n\032\n"):
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith(b"\377\330"):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input_f.seek(0)
            input_f.read(2)
            b_ = input_f.read(1)
            try:
                while (b_ and b_[0] != 0xDA):
                    while (b_[0] != 0xFF):
                        b_ = input_f.read(1)
                    while (b_[0] == 0xFF):
                        b_ = input_f.read(1)
                    if (0xC0 <= b_[0] <= 0xC3):
                        input_f.read(3)
                        h, w = struct.unpack(">HH", input_f.read(4))
                        break
                    else:
                        seg_len = struct.unpack(">H", input_f.read(2))[0]
                        input_f.read(seg_len - 2)
                    b_ = input_f.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height


def getGroupSize(channels):
    if channels>=32:
        goalSize=8
    else:
        goalSize=4
    if channels%goalSize==0:
        return goalSize
    factors=primeFactors(channels)
    bestDist=9999
    for f in factors:
        if abs(f-goalSize)<=bestDist: #favor larger
            bestDist=abs(f-goalSize)
            bestGroup=f
    return int(bestGroup)

