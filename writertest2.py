import cv2
import Levenshtein as Lev
import random
import numpy as np
import torch
from network_tro import ConTranModel
from load_data import (
    IMG_HEIGHT, IMG_WIDTH, NUM_WRITERS,
    letter2index, tokens, num_tokens, OUTPUT_MAX_LEN, index2letter
)
from modules_tro import normalize
import os
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# ----- Paths you already use -----
folder_wids = '/home/woody/iwi5/iwi5333h/data'
img_base = '/home/woody/iwi5/iwi5333h/data'  # IAM-style: a01/a01-000u/a01-000u-00.png
folder_pre = '/home/woody/iwi5/iwi5333h/checkwords2/'
epoch_default = 5000

# ----- CLI -----
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--epoch', default=epoch_default, type=int,
    help=('Epoch/model checkpoint number, used to pick model path '
          'like contran-<epoch>.model')
)
parser.add_argument(
    '--writers',
    nargs='+',
    default=None,
    help="One or more writer IDs to process (e.g., a01 a02). If omitted, all writers in the target file are processed."
)
parser.add_argument(
    '--pair_file',
    type=str,
    default=None,
    help='Text file with lines like "writer_id,word" (or "writer_id word"). '
         'If provided, overrides the corpus for those writers.'
)

# ----- Utilities -----
def pre_data(data_dict, target_file):
    """Build {writer_id: [[imgname, label], ...]} from a target file:
       lines like: 'a01,a01-000u-00 label' """
    with open(target_file, 'r') as _f:
        data = _f.readlines()
        labels = [i.split(' ')[1].replace('\n', '').replace('\r', '') for i in data]
        data = [i.split(' ')[0] for i in data]
        wids = [i.split(',')[0] for i in data]
        imgnames = [i.split(',')[1] for i in data]

    for wid, imgname, label in zip(wids, imgnames, labels):
        data_dict.setdefault(wid, []).append([imgname, label])
    return data_dict


def load_pairs(path):
    """Read writer–word pairs. Supports 'wid,word' or 'wid word'."""
    wid2words = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ',' in line:
                wid, word = line.split(',', 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                wid, word = parts
            wid = wid.strip()
            word = word.strip()
            if wid and word:
                wid2words.setdefault(wid, []).append(word)
    return wid2words


gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _read_image(file_name, thresh=None):
    subfolder = file_name.split('-')[0]         # e.g., 'a01'
    parent = '-'.join(file_name.split('-')[:2]) # e.g., 'a01-000u'
    url = os.path.join(img_base, subfolder, parent, file_name + '.png')

    if not os.path.exists(url):
        print(f"⚠️ Image not found: {url}")
        return None

    img = cv2.imread(url, 0)
    if img is None:
        print(f"⚠️ Failed to read image (cv2.imread returned None): {url}")
        return None

    if thresh:
        pass

    # scale to IMG_HEIGHT, keep aspect
    rate = float(IMG_HEIGHT) / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * rate) + 1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = img / 255.0  # [0,1]
    img = 1.0 - img    # invert
    img_width = img.shape[-1]

    # pad/crop to IMG_WIDTH
    if img_width > IMG_WIDTH:
        outImg = img[:, :IMG_WIDTH]
    else:
        outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
        outImg[:, :img_width] = img
    outImg = outImg.astype('float32')

    # normalize
    mean = 0.5
    std = 0.5
    outImgFinal = (outImg - mean) / std
    return outImgFinal


def _to_label_vec(word: str) -> np.ndarray:
    """Convert a string to a fixed-length int64 vector with special tokens/padding."""
    indices = []
    for ch in word:
        if ch not in letter2index:
            raise KeyError(ch)
        indices.append(letter2index[ch])
    arr = np.asarray(indices, dtype=np.int64) + np.int64(num_tokens)
    arr = np.concatenate(([tokens['GO_TOKEN']], arr, [tokens['END_TOKEN']])).astype(np.int64)
    if len(arr) < OUTPUT_MAX_LEN:
        pad = np.full(OUTPUT_MAX_LEN - len(arr), tokens['PAD_TOKEN'], dtype=np.int64)
        arr = np.concatenate((arr, pad)).astype(np.int64)
    else:
        arr = arr[:OUTPUT_MAX_LEN].astype(np.int64)
    return arr


def test_writer(wid, model_file, folder, texts_or_path, data_dict, is_list=False):
    """Generate images for a single writer.
       texts_or_path: list[str] if is_list=True, else path to corpus file."""
    # ----- collect ~50 style images for this writer -----
    if wid not in data_dict:
        print(f"⚠️ Writer {wid} not found in target_file index. Skipping.")
        return

    raw_items = data_dict[wid]
    imgs_list = []
    for item in raw_items:
        im = _read_image(item[0])
        if im is not None:
            imgs_list.append(im)

    if len(imgs_list) == 0:
        print(f"⚠️ No valid images for writer {wid}, skipping.")
        return

    random.shuffle(imgs_list)
    final_imgs = imgs_list[:50]
    if len(final_imgs) < 50:
        # repeat from available to reach 50
        while len(final_imgs) < 50 and len(imgs_list) > 0:
            num_cp = min(50 - len(final_imgs), len(imgs_list))
            final_imgs.extend(imgs_list[:num_cp])

    imgs = torch.from_numpy(np.array(final_imgs)).unsqueeze(0).to(gpu)  # (1, 50, H, W)

    # ----- load labels (from list or file) -----
    if is_list:
        texts = [t.strip() for t in texts_or_path if t.strip()]
    else:
        with open(texts_or_path, 'r') as _f:
            texts = [t.strip() for t in _f.read().split() if t.strip()]

    if len(texts) == 0:
        print(f"⚠️ No texts provided for writer {wid}. Skipping.")
        return

    # Convert to fixed-length int64 vectors; skip OOV words safely
    label_vecs = []
    oov_words = []
    for tx in texts:
        try:
            label_vecs.append(_to_label_vec(tx))
        except KeyError as e:
            oov_words.append((tx, str(e)))
            continue

    if oov_words:
        print(f"ℹ️ Writer {wid}: skipping {len(oov_words)} word(s) with OOV char(s): "
              f"{', '.join([w for w,_ in oov_words[:5]])}{' ...' if len(oov_words) > 5 else ''}")

    if not label_vecs:
        print(f"⚠️ Writer {wid}: no valid words after filtering. Skipping.")
        return

    labels_np = np.stack(label_vecs).astype(np.int64)
    labels = torch.from_numpy(labels_np).to(gpu)

    # ----- model -----
    model = ConTranModel(NUM_WRITERS, 0, True).to(gpu)
    model.load_state_dict(torch.load(model_file, map_location=gpu))
    model.eval()

    # ----- generate -----
    num_saved = 0
    with torch.no_grad():
        f_xss = model.gen.enc_image(imgs)
        f_xs = f_xss[-1]
        for label in labels:
            label = label.unsqueeze(0)
            f_xt, f_embed = model.gen.enc_text(label, f_xs.shape)
            f_mix = model.gen.mix(f_xss, f_embed)
            xg = model.gen.decode(f_mix, f_xss, f_embed, f_xt)
            pred = model.rec(xg, label, img_width=torch.from_numpy(np.array([IMG_WIDTH])))

            # decode tokens back to strings
            label_seq = label.squeeze().detach().cpu().numpy().tolist()
            pred_seq = torch.topk(pred, 1, dim=-1)[1].squeeze().detach().cpu().numpy().tolist()

            # strip special tokens (IDs < num_tokens)
            for j in range(num_tokens):
                label_seq = list(filter(lambda x: x != j, label_seq))
                pred_seq = list(filter(lambda x: x != j, pred_seq))

            label_str = ''.join([index2letter[c - num_tokens] for c in label_seq])
            pred_str  = ''.join([index2letter[c - num_tokens] for c in pred_seq])

            ed_value = Lev.distance(pred_str, label_str)

            # save if acceptable (keep your original ≤100 guard)
            if ed_value <= 100:
                num_saved += 1
                xg_np = xg.detach().cpu().numpy().squeeze()
                xg_np = normalize(xg_np)
                xg_np = 255 - xg_np

                os.makedirs(folder, exist_ok=True)
                out_path = os.path.join(folder, f'{wid}-{num_saved}.{label_str}-{pred_str}.png')
                ok = cv2.imwrite(out_path, xg_np)
                if not ok:
                    print(f"⚠️ Failed to save {out_path}")

    print(f'Writer {wid}: saved {num_saved} images.')


if __name__ == '__main__':
    args = parser.parse_args()
    model_epoch = str(args.epoch)

    # ---- scenario 0 (your active one) ----
    folder = os.path.join(folder_pre, model_epoch, 'res_4.oo_vocab_te_writer')
    target_file = '/home/woody/iwi5/iwi5333h/AFFGanWriting/Groundtruth/gan.iam.tr_va.gt.filter27'
    text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/corpora_english/writerword.57'

    os.makedirs(folder, exist_ok=True)
    print(f'Output folder: {folder}')

    # index dataset once
    data_dict = pre_data(dict(), target_file)
    model_path = f'/home/vault/iwi5/iwi5333h/bestmodel/contran-{model_epoch}.model'

    # ---- mixed mode: specific words per writer via --pair_file ----
    if args.pair_file:
        wid2words = load_pairs(args.pair_file)

        # optional filter with --writers
        if args.writers:
            requested = set(args.writers)
            before = set(wid2words.keys())
            wid2words = {w: ws for w, ws in wid2words.items() if w in requested}
            missing = requested - set(wid2words.keys())
            if missing:
                print(f"⚠️ Requested writer(s) not present in pair_file: {sorted(missing)}")
            if not wid2words:
                print("⚠️ No writers to process after filtering; exiting.")
                raise SystemExit(0)

        wids = tqdm(list(wid2words.keys()), desc='Writers (pair mode)')
        for wid in wids:
            test_writer(wid, model_path, folder, wid2words[wid], data_dict, is_list=True)

    # ---- original mode: one corpus for each selected writer ----
    else:
        # Build the set of all writers from the target file
        with open(target_file, 'r') as _f:
            data_lines = _f.readlines()
        all_wids = list(set([i.split(',')[0] for i in data_lines]))

        if args.writers:
            requested = set(args.writers)
            available = set(all_wids)
            missing = requested - available
            if missing:
                print(f"⚠️ Requested writer(s) not found in {target_file}: {sorted(missing)}")
            wids = [w for w in all_wids if w in requested]
        else:
            wids = all_wids

        wids = tqdm(wids, desc='Writers (corpus mode)')
        for wid in wids:
            test_writer(wid, model_path, folder, text_corpus, data_dict, is_list=False)
