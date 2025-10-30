# datasets/author_hw_dataset.py
import json
from glob import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from collections import defaultdict
import os
import cv2
import numpy as np
import math
import itertools
import pickle
import random

from utils import grid_distortion
from utils.util import ensure_dir, makeMask
from utils import string_utils, augmentation, normalize_line
from utils.parseIAM import getLineBoundaries as parseXML


# ---------- helpers ----------

def resolve_form_path(root: str, name: str) -> str:
    """Find the correct IAM form image path under root."""
    bases = [name, f"{name}u", f"{name}x"]
    exts  = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
    dirs  = ["forms", "formsA-D", "formsE-H", "formsI-Z"]
    subdirs = ["", name[:3]]  # some dumps use a subfolder like c06/

    for d in dirs:
        for sd in subdirs:
            base_dir = Path(root) / d / sd if sd else Path(root) / d
            for b in bases:
                for ext in exts:
                    p = base_dir / f"{b}{ext}"
                    if p.exists():
                        return str(p)
    raise FileNotFoundError(f"Form image not found for {name} under {root}.")


PADDING_CONSTANT = -1

def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


# ---------- collate ----------

def collate(batch):
    # Filter failed samples first
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise RuntimeError("All samples in this batch failed to load.")
    if len(batch) == 1:
        batch[0]['a_batch_size'] = batch[0]['image'].size(0)
        return batch[0]

    a_batch_size = len(batch[0]['gt'])

    dim1 = batch[0]['image'].shape[1]
    dim3 = max([b['image'].shape[3] for b in batch])
    dim2 = batch[0]['image'].shape[2]

    max_label_len = max([b['label'].size(0) for b in batch])
    if batch[0]['spaced_label'] is not None:
        max_spaced_label_len = max([b['spaced_label'].size(0) for b in batch])
    else:
        max_spaced_label_len = None

    input_batch = torch.FloatTensor(len(batch) * a_batch_size, dim1, dim2, dim3).fill_(PADDING_CONSTANT)
    mask_batch = torch.FloatTensor(len(batch) * a_batch_size, dim1, dim2, dim3).fill_(PADDING_CONSTANT)
    if 'fg_mask' in batch[0]:
        fg_masks = torch.FloatTensor(len(batch) * a_batch_size, 1, dim2, dim3).fill_(0)
    if 'changed_image' in batch[0]:
        changed_batch = torch.FloatTensor(len(batch) * a_batch_size, dim1, dim2, dim3).fill_(PADDING_CONSTANT)
    top_and_bottom_batch = torch.FloatTensor(len(batch) * a_batch_size, 2, dim3).fill_(0)
    center_line_batch = torch.FloatTensor(len(batch) * a_batch_size, dim3).fill_(dim2 / 2)
    labels_batch = torch.IntTensor(max_label_len, len(batch) * a_batch_size).fill_(0)
    if max_spaced_label_len is not None:
        spaced_labels_batch = torch.IntTensor(max_spaced_label_len, len(batch) * a_batch_size).fill_(0)
    else:
        spaced_labels_batch = None

    for i in range(len(batch)):
        b_img = batch[i]['image']
        b_mask = batch[i]['mask']
        b_top_and_bottom = batch[i]['top_and_bottom']
        b_center_line = batch[i]['center_line']
        l = batch[i]['label']

        input_batch[i * a_batch_size:(i + 1) * a_batch_size, :, :, 0:b_img.shape[3]] = b_img
        mask_batch[i * a_batch_size:(i + 1) * a_batch_size, :, :, 0:b_img.shape[3]] = b_mask
        if 'fg_mask' in batch[i]:
            fg_masks[i * a_batch_size:(i + 1) * a_batch_size, :, :, 0:b_img.shape[3]] = batch[i]['fg_mask']
        if 'changed_image' in batch[i]:
            changed_batch[i * a_batch_size:(i + 1) * a_batch_size, :, :, 0:b_img.shape[3]] = batch[i]['changed_image']

        if b_top_and_bottom is not None:
            top_and_bottom_batch[i * a_batch_size:(i + 1) * a_batch_size, :, 0:b_img.shape[3]] = b_top_and_bottom
        else:
            top_and_bottom_batch = None

        if b_center_line is not None:
            center_line_batch[i * a_batch_size:(i + 1) * a_batch_size, 0:b_img.shape[3]] = b_center_line
        else:
            center_line_batch = None

        labels_batch[0:l.size(0), i * a_batch_size:(i + 1) * a_batch_size] = l
        if max_spaced_label_len is not None:
            sl = batch[i]['spaced_label']
            spaced_labels_batch[0:sl.size(0), i * a_batch_size:(i + 1) * a_batch_size] = sl

    style = None if batch[0]['style'] is None else torch.cat([b['style'] for b in batch], dim=0)

    return {
        "image": input_batch,
        "mask": mask_batch,
        "top_and_bottom": top_and_bottom_batch,
        "center_line": center_line_batch,
        "label": labels_batch,
        "style": style,
        "label_lengths": torch.cat([b['label_lengths'] for b in batch], dim=0),
        "gt": [l for b in batch for l in b['gt']],
        "spaced_label": spaced_labels_batch,
        "author": [l for b in batch for l in b['author']],
        "author_idx": [l for b in batch for l in b['author_idx']],
        "name": [l for b in batch for l in b['name']],
        "a_batch_size": a_batch_size,
        **({"fg_mask": fg_masks} if 'fg_mask' in batch[0] else {}),
        **({"changed_image": changed_batch} if 'changed_image' in batch[0] else {}),
    }


# ---------- dataset ----------

class AuthorHWDataset(Dataset):
    def __init__(self, dirPath, split, config):
        if 'split' in config:
            split = config['split']

        self.img_height = config['img_height']
        self.batch_size = config['a_batch_size']
        self.no_spaces = config.get('no_spaces', False)
        self.max_width = config.get('max_width', 3000)
        self.warning = False

        self.triplet = config.get('triplet', False)
        if self.triplet:
            self.triplet_author_size = config['triplet_author_size']
            self.triplet_sample_size = config['triplet_sample_size']

        only_author = config.get('only_author', None)
        skip_author = config.get('skip_author', None)

        with open(os.path.join('data', 'sets.json')) as f:
            set_list = json.load(f)[split]

        self.authors = defaultdict(list)
        self.lineIndex = []
        self.max_char_len = 0
        self.author_list = set()

        for page_idx, name in enumerate(set_list):
            lines, author = parseXML(os.path.join(dirPath, 'xmls', name + '.xml'))
            self.author_list.add(author)
            if only_author is not None and isinstance(only_author, int) and page_idx == only_author:
                only_author = author
                print(f'Only author: {only_author}')
            if only_author is not None and author != only_author:
                continue
            if skip_author is not None and author == skip_author:
                continue

            self.max_char_len = max([self.max_char_len] + [len(l[1]) for l in lines])

            img_path = resolve_form_path(dirPath, name)
            self.authors[author] += [(img_path,) + l for l in lines]

        self.author_list = sorted(list(self.author_list))

        short = config.get('short', False)
        for author, lines in self.authors.items():
            for i in range(len(lines) // self.batch_size):
                ls = [self.batch_size * i + n for n in range(self.batch_size)]
                self.lineIndex.append((author, ls))
                if short and i >= short:
                    break
            if short and i >= short:
                continue
            leftover = len(lines) % self.batch_size
            fill = self.batch_size - leftover
            last = list(range(fill)) + [len(lines) - (1 + i) for i in range(leftover)]
            self.lineIndex.append((author, last))

        self.fg_masks_dir = config.get('fg_masks_dir', None)
        if self.fg_masks_dir is not None:
            if self.fg_masks_dir.endswith('/'):
                self.fg_masks_dir = self.fg_masks_dir[:-1]
            self.fg_masks_dir += f'_{self.max_width}'
            ensure_dir(self.fg_masks_dir)
            for author, lines in self.lineIndex:
                for line in lines:
                    img_path, lb, gt = self.authors[author][line]
                    fg_path = os.path.join(self.fg_masks_dir, f'{author}_{line}.png')
                    if not os.path.exists(fg_path):
                        img = cv2.imread(img_path, 0)[lb[0]:lb[1], lb[2]:lb[3]]
                        if img.shape[0] != self.img_height:
                            if img.shape[0] < self.img_height and not self.warning:
                                self.warning = True
                                print("WARNING: upsampling image to fit size")
                            percent = float(self.img_height) / img.shape[0]
                            if img.shape[1] * percent > self.max_width:
                                percent = self.max_width / img.shape[1]
                            img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)
                            if img.shape[0] < self.img_height:
                                diff = self.img_height - img.shape[0]
                                img = np.pad(img, ((diff // 2, diff // 2 + diff % 2), (0, 0)),
                                             'constant', constant_values=255)
                        th, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        binarized = 255 - binarized
                        ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                        binarized = cv2.dilate(binarized, ele)
                        cv2.imwrite(fg_path, binarized)
                        print(f'saved fg mask: {fg_path}')

        char_set_path = config['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']

        self.augmentation = config.get('augmentation', None)
        self.normalized_dir = config.get('cache_normalized', None)
        if self.normalized_dir is not None:
            ensure_dir(self.normalized_dir)
        self.max_strech = 0.4
        self.max_rot_rad = 45 / 180 * math.pi

        self.remove_bg = config.get('remove_bg', False)
        self.include_stroke_aug = config.get('include_stroke_aug', False)

        if config.get('overfit', False):
            self.lineIndex = self.lineIndex[:10]

        self.center = False

        if 'style_loc' in config:
            by_author_styles = defaultdict(list)
            by_author_all_ids = defaultdict(set)
            style_loc = config['style_loc']
            if not style_loc.endswith('*'):
                style_loc += '*'
            all_style_files = glob(style_loc)
            assert len(all_style_files) > 0
            for loc in all_style_files:
                with open(loc, 'rb') as f:
                    styles = pickle.load(f)
                for i in range(len(styles['authors'])):
                    by_author_styles[styles['authors'][i]].append((styles['styles'][i], styles['ids'][i]))
                    by_author_all_ids[styles['authors'][i]].update(styles['ids'][i])

            self.styles = defaultdict(lambda: defaultdict(list))
            for author in by_author_styles:
                for id_ in by_author_all_ids[author]:
                    for style, ids in by_author_styles[author]:
                        if id_ not in ids:
                            self.styles[author][id_].append(style)
            for author in self.authors:
                assert author in self.styles
        else:
            self.styles = None

        if 'spaced_loc' in config:
            with open(config['spaced_loc'], 'rb') as f:
                self.spaced_by_name = pickle.load(f)
        else:
            self.spaced_by_name = None
            self.identity_spaced = config.get('no_spacing_for_spaced', False)

        self.mask_post = config.get('mask_post', [])
        self.mask_random = config.get('mask_random', False)

    def __len__(self):
        return len(self.lineIndex)

    def __getitem__(self, idx):
        if isinstance(self.augmentation, str) and 'affine' in self.augmentation:
            strech = (self.max_strech * 2) * np.random.random() - self.max_strech + 1
            skew = (self.max_rot_rad * 2) * np.random.random() - self.max_rot_rad
        if self.include_stroke_aug:
            thickness_change = np.random.randint(-4, 5)
            fg_shade = np.random.random() * 0.25 + 0.75
            bg_shade = np.random.random() * 0.2
            blur_size = np.random.randint(2, 4)
            noise_sigma = np.random.random() * 0.02

        batch = []

        if self.triplet == 'hard':
            authors = random.sample(self.authors.keys(), self.triplet_author_size)
            alines = []
            for author in authors:
                if len(self.authors[author]) >= self.triplet_sample_size * self.batch_size:
                    lines = random.sample(range(len(self.authors[author])),
                                         self.triplet_sample_size * self.batch_size)
                else:
                    lines = list(range(len(self.authors[author])))
                    random.shuffle(lines)
                    dif = self.triplet_sample_size * self.batch_size - len(self.authors[author])
                    lines += lines[:dif]
                alines += [(author, l) for l in lines]
        else:
            inst = self.lineIndex[idx]
            author = inst[0]
            lines = inst[1]
            alines = [(author, l) for l in lines]
            used_lines = set(lines)
            if self.triplet:
                if len(self.authors[author]) <= 2 * self.batch_size:
                    for l in range(len(self.authors[author])):
                        if l not in used_lines:
                            alines.append((author, l))
                    if len(alines) < 2 * self.batch_size:
                        dif = 2 * self.batch_size - len(alines)
                        for i in range(dif):
                            alines.append(alines[self.batch_size + i])
                else:
                    unused_lines = set(range(len(self.authors[author]))) - used_lines
                    for i in range(self.batch_size):
                        l = random.choice(list(unused_lines))
                        unused_lines.remove(l)
                        alines.append((author, l))

                other_authors = set(range(len(self.authors)))
                other_authors.remove(author)
                author = random.choice(list(other_authors))
                unused_lines = set(range(len(self.authors[author]))) - used_lines
                for i in range(self.batch_size):
                    l = random.choice(list(unused_lines))
                    unused_lines.remove(l)
                    alines.append((author, l))

        # read / pre-process each line
        image_tuples = []
        for author, line in alines:
            if line >= len(self.authors[author]):
                line = (line + 37) % len(self.authors[author])
            img_path, lb, gt = self.authors[author][line]

            if self.no_spaces:
                gt = gt.replace(' ', '')
            if isinstance(self.augmentation, str) and 'normalization' in self.augmentation and \
               self.normalized_dir is not None and os.path.exists(os.path.join(self.normalized_dir, f'{author}_{line}.png')):
                img = cv2.imread(os.path.join(self.normalized_dir, f'{author}_{line}.png'), 0)
                readNorm = True
            else:
                img = cv2.imread(img_path, 0)
                if img is None:
                    print(f'Error, could not read image: {img_path}')
                    return None
                img = img[lb[0]:lb[1], lb[2]:lb[3]]
                readNorm = False

            if img.shape[0] != self.img_height:
                if img.shape[0] < self.img_height and not self.warning:
                    self.warning = True
                    print("WARNING: upsampling image to fit size")
                percent = float(self.img_height) / img.shape[0]
                if img.shape[1] * percent > self.max_width:
                    percent = self.max_width / img.shape[1]
                img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)
                if img.shape[0] < self.img_height:
                    diff = self.img_height - img.shape[0]
                    img = np.pad(img, ((diff // 2, diff // 2 + diff % 2), (0, 0)), 'constant', constant_values=255)
            elif img.shape[1] > self.max_width:
                percent = self.max_width / img.shape[1]
                img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)
                if img.shape[0] < self.img_height:
                    diff = self.img_height - img.shape[0]
                    img = np.pad(img, ((diff // 2, diff // 2 + diff % 2), (0, 0)), 'constant', constant_values=255)

            if self.augmentation == 'affine' and img.shape[1] * strech > self.max_width:
                strech = self.max_width / img.shape[1]

            image_tuples.append((line, gt, img, author, readNorm))

        for line, gt, img, author, readNorm in image_tuples:
            if self.fg_masks_dir is not None:
                fg_path = os.path.join(self.fg_masks_dir, f'{author}_{line}.png')
                fg_mask = cv2.imread(fg_path, 0)
                fg_mask = fg_mask / 255
                if fg_mask.shape != img.shape:
                    print(f'Error, fg_mask ({fg_path}, {fg_mask.shape}) not same size as image ({img.shape})')
                    th, fg_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    fg_mask = 255 - fg_mask
                    ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    fg_mask = cv2.dilate(fg_mask, ele)
                    fg_mask = fg_mask / 255
            else:
                fg_mask = None

            if isinstance(self.augmentation, str) and 'normalization' in self.augmentation and not readNorm:
                img = normalize_line.deskew(img)
                img = normalize_line.skeletonize(img)
                if self.normalized_dir is not None:
                    cv2.imwrite(os.path.join(self.normalized_dir, f'{author}_{line}.png'), img)
            if isinstance(self.augmentation, str) and 'affine' in self.augmentation:
                img, fg_mask = augmentation.affine_trans(img, fg_mask, skew, strech)
            elif self.augmentation is not None and (type(self.augmentation) is not None or 'warp' in self.augmentation):
                img = augmentation.apply_tensmeyer_brightness(img)
                img = grid_distortion.warp_image(img)
                assert fg_mask is None

            if self.include_stroke_aug:
                new_img = augmentation.change_thickness(img, thickness_change, fg_shade, bg_shade, blur_size, noise_sigma)
                if len(new_img.shape) == 2:
                    new_img = new_img[..., None]
                new_img = new_img * 2 - 1.0

            if len(img.shape) == 2:
                img = img[..., None]

            img = img.astype(np.float32)
            if self.remove_bg:
                img = 1.0 - img / 256.0
                blurred_mask = cv2.blur(fg_mask, (7, 7))
                img *= blurred_mask[..., None]
                img = 2 * img - 1
            else:
                img = 1.0 - img / 128.0

            if len(gt) == 0:
                return None
            gt_label = string_utils.str2label_single(gt, self.char_to_idx)

            style = None  # keep simple unless you wire styles back in

            name = f'{author}_{line}'
            if getattr(self, 'identity_spaced', False):
                spaced_label = gt_label[:, None].astype(np.int64)
            else:
                spaced_label = None if self.spaced_by_name is None else self.spaced_by_name.get(name)
                if spaced_label is not None:
                    assert spaced_label.shape[1] == 1

            entry = {
                "image": img,
                "gt": gt,
                "style": style,
                "gt_label": gt_label,
                "spaced_label": spaced_label,
                "name": name,
                "center": self.center,
                "author": author,
                "author_idx": self.author_list.index(author),
            }
            if self.fg_masks_dir is not None:
                entry['fg_mask'] = fg_mask
            if self.include_stroke_aug:
                entry['changed_image'] = new_img
            batch.append(entry)

        # sanity
        assert len(set([b['image'].shape[0] for b in batch])) == 1
        assert len(set([b['image'].shape[2] for b in batch])) == 1

        dim0 = batch[0]['image'].shape[0]
        dim1 = max([b['image'].shape[1] for b in batch])
        dim2 = batch[0]['image'].shape[2]

        all_labels = []
        label_lengths = []
        if self.spaced_by_name is not None or getattr(self, 'identity_spaced', False):
            spaced_labels = []
        else:
            spaced_labels = None
        max_spaced_len = 0

        input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT, dtype=np.float32)
        if self.fg_masks_dir is not None:
            fg_masks = np.full((len(batch), dim0, dim1, 1), 0, dtype=np.float32)
        if self.include_stroke_aug:
            changed_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT, dtype=np.float32)

        for i in range(len(batch)):
            b_img = batch[i]['image']
            toPad = (dim1 - b_img.shape[1])
            if batch[0].get('center', False):
                toPad //= 2
            else:
                toPad = 0
            input_batch[i, :, toPad:toPad + b_img.shape[1], :] = b_img
            if self.fg_masks_dir is not None:
                fg_masks[i, :, toPad:toPad + b_img.shape[1], 0] = batch[i]['fg_mask']
            if self.include_stroke_aug:
                changed_batch[i, :, toPad:toPad + b_img.shape[1], :] = batch[i]['changed_image']

            l = batch[i]['gt_label']
            all_labels.append(l)
            label_lengths.append(len(l))

            if spaced_labels is not None:
                sl = batch[i]['spaced_label']
                if sl is not None:
                    spaced_labels.append(sl)
                    max_spaced_len = max(max_spaced_len, sl.shape[0])

        # labels: pad → stack → torch
        max_len = max(label_lengths)                       # Python int
        all_labels = [np.asarray(l, dtype=np.int32) for l in all_labels]
        all_labels = [np.pad(l, (0, max_len - l.shape[0]), 'constant') for l in all_labels]
        all_labels = np.stack(all_labels, axis=1)          # (T, B)
        labels = torch.tensor(all_labels, dtype=torch.int32)
        label_lengths = torch.IntTensor(label_lengths)

        if spaced_labels is not None and len(spaced_labels) > 0:
            spaced_labels = [np.pad(l, ((0, max_spaced_len - l.shape[0]), (0, 0)), 'constant') for l in spaced_labels]
            spaced_labels = np.concatenate(spaced_labels, axis=1)
            spaced_labels = torch.tensor(spaced_labels, dtype=torch.int32)

        # images/masks: numpy → torch, and makeMask on numpy
        images_np = np.asarray(input_batch.transpose(0, 3, 1, 2), dtype=np.float32)
        images_np = np.ascontiguousarray(images_np)

        # makeMask expects numpy
        mask_np, top_and_bottom_np, center_line_np = makeMask(images_np, self.mask_post, self.mask_random)
        mask = torch.tensor(mask_np, dtype=torch.float32)
        top_and_bottom = (torch.tensor(top_and_bottom_np, dtype=torch.float32)
                          if top_and_bottom_np is not None else None)
        center_line = (torch.tensor(center_line_np, dtype=torch.float32)
                       if center_line_np is not None else None)

        images = torch.tensor(images_np, dtype=torch.float32)

        if self.fg_masks_dir is not None:
            fg_masks_np = np.asarray(fg_masks.transpose(0, 3, 1, 2), dtype=np.float32)
            fg_masks_np = np.ascontiguousarray(fg_masks_np)
            fg_masks = torch.tensor(fg_masks_np, dtype=torch.float32)

        if self.include_stroke_aug:
            changed_np = np.asarray(changed_batch.transpose(0, 3, 1, 2), dtype=np.float32)
            changed_np = np.ascontiguousarray(changed_np)
            changed_images = torch.tensor(changed_np, dtype=torch.float32)

        toRet = {
            "image": images,
            "mask": mask,
            "top_and_bottom": top_and_bottom,
            "center_line": center_line,
            "label": labels,
            "style": None,  # wire styles back in if needed
            "label_lengths": label_lengths,
            "gt": [b['gt'] for b in batch],
            "spaced_label": spaced_labels,
            "name": [b['name'] for b in batch],
            "author": [b['author'] for b in batch],
            "author_idx": [b['author_idx'] for b in batch],
        }
        if self.fg_masks_dir is not None:
            toRet['fg_mask'] = fg_masks
        if self.include_stroke_aug:
            toRet['changed_image'] = changed_images

        return toRet

    def max_len(self):
        return self.max_char_len
