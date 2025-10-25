# generate_by_writer.py
import cv2
import Levenshtein as Lev
import random
import numpy as np
import torch
from network_tro import ConTranModel
from load_data import (
    IMG_HEIGHT, IMG_WIDTH, NUM_WRITERS, letter2index, tokens, num_tokens,
    OUTPUT_MAX_LEN, index2letter
)
from modules_tro import normalize
import os
import sys
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# --------- Paths you may want to adjust ----------
IMG_BASE_DEFAULT = '/home/woody/iwi5/iwi5333h/data'
OUT_BASE_DEFAULT = '/home/woody/iwi5/iwi5333h/checkwords'
MODEL_DEFAULT    = '/home/vault/iwi5/iwi5333h/thebest/contran-{epoch}.model'
# -------------------------------------------------

gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def label_padding(label_str: str):
    ll = [letter2index[c] for c in label_str]
    ll = np.array(ll) + num_tokens
    ll = [tokens['GO_TOKEN']] + list(ll) + [tokens['END_TOKEN']]
    # pad
    if len(ll) < OUTPUT_MAX_LEN:
        ll.extend([tokens['PAD_TOKEN']] * (OUTPUT_MAX_LEN - len(ll)))
    return ll

def read_image(path: str):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, 0)
    if img is None:
        return None
    rate = float(IMG_HEIGHT) / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * rate) + 1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = 1.0 - (img / 255.0)

    img_width = img.shape[-1]
    if img_width > IMG_WIDTH:
        out = img[:, :IMG_WIDTH]
    else:
        out = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
        out[:, :img_width] = img
    out = out.astype('float32')
    # normalize to match training
    mean, std = 0.5, 0.5
    return (out - mean) / std

def discover_images_for_writer(img_base: str, wid: str):
    """
    Looks under img_base/<wid>/** for .png files.
    This covers IAM-like layouts if each writer has a top-level folder named by wid.
    """
    root = os.path.join(img_base, wid)
    paths = []
    if os.path.isdir(root):
        for r, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith('.png'):
                    paths.append(os.path.join(r, f))
    # Fallback: scan entire tree (slower) for filenames that begin with wid + '-'
    if not paths:
        for r, _, files in os.walk(img_base):
            for f in files:
                if f.lower().endswith('.png') and (f.startswith(wid + '-') or f.startswith(wid)):
                    paths.append(os.path.join(r, f))
    return paths

def load_words_per_writer(target_file: str):
    """
    Reads CSV/CSV-like lines: writerid,label
    Header ('writerid,label') is optional and ignored.
    Returns dict: { wid -> [word1, word2, ...] }
    """
    mapping = {}
    with open(target_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # tolerate accidental spaces + split once on comma
            # format: wid,label   (label may contain commas rarely; if so, keep first split)
            if ',' not in line:
                continue
            wid, label = line.split(',', 1)
            wid = wid.strip()
            label = label.strip()
            # skip header row
            if wid.lower() in {'writerid', 'writer', 'id'}:
                continue
            if not wid or not label:
                continue
            mapping.setdefault(wid, []).append(label)
    return mapping

def build_batch_of_images(img_paths, max_n=50):
    """Collect up to max_n images, repeat if necessary to reach max_n (like your original script)."""
    valid = []
    for p in img_paths:
        im = read_image(p)
        if im is not None:
            valid.append(im)
    if not valid:
        return None
    random.shuffle(valid)
    final_imgs = valid[:max_n]
    # top-up (repeat) if fewer than max_n available
    while len(final_imgs) < max_n:
        need = max_n - len(final_imgs)
        final_imgs += valid[:min(need, len(valid))]
    arr = torch.from_numpy(np.array(final_imgs)).unsqueeze(0).to(gpu)  # 1, N, H, W
    return arr

def generate_for_writer(wid, model_file, out_folder, words, img_base):
    img_paths = discover_images_for_writer(img_base, wid)
    if not img_paths:
        print(f"⚠️ No images found for writer {wid} under {img_base}. Skipping.")
        return

    imgs = build_batch_of_images(img_paths, max_n=50)
    if imgs is None:
        print(f"⚠️ Could not build image batch for writer {wid}. Skipping.")
        return

    labels_np = np.array([np.array(label_padding(w)) for w in words], dtype=np.int64)
    labels = torch.from_numpy(labels_np).to(gpu)

    # model
    model = ConTranModel(NUM_WRITERS, 0, True).to(gpu)
    model.load_state_dict(torch.load(model_file, map_location=gpu))
    model.eval()

    os.makedirs(out_folder, exist_ok=True)

    with torch.no_grad():
        f_xss = model.gen.enc_image(imgs)
        f_xs = f_xss[-1]
        saved = 0
        for label in labels:
            label = label.unsqueeze(0)
            f_xt, f_embed = model.gen.enc_text(label, f_xs.shape)
            f_mix = model.gen.mix(f_xss, f_embed)
            xg = model.gen.decode(f_mix, f_xss, f_embed, f_xt)
            pred = model.rec(xg, label, img_width=torch.from_numpy(np.array([IMG_WIDTH])))

            # decode tokens -> strings
            label_list = label.squeeze().cpu().numpy().tolist()
            pred_ids = torch.topk(pred, 1, dim=-1)[1].squeeze().cpu().numpy().tolist()

            for j in range(num_tokens):
                label_list = list(filter(lambda x: x != j, label_list))
                pred_ids   = list(filter(lambda x: x != j, pred_ids))

            gt  = ''.join([index2letter[c - num_tokens] for c in label_list])
            hyp = ''.join([index2letter[c - num_tokens] for c in pred_ids])

            if Lev.distance(hyp, gt) <= 100:  # keep your original relaxed gate
                saved += 1
                xg_np = xg.cpu().numpy().squeeze()
                xg_np = normalize(xg_np)
                xg_np = 255 - xg_np
                out_path = os.path.join(out_folder, f'{wid}-{saved}.{gt}-{hyp}.png')
                cv2.imwrite(out_path, xg_np)

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', type=int, default=3600,
                        help='Epoch/model checkpoint number, used to pick model path like contran-<epoch>.model')
    parser.add_argument('--target_file', required=True,
                        help='Path to CSV-like file with rows: writerid,label (header optional)')
    parser.add_argument('--img_base', default=IMG_BASE_DEFAULT,
                        help='Root directory of images. Script will look under <img_base>/<writer_id>/**.png')
    parser.add_argument('--out_base', default=OUT_BASE_DEFAULT,
                        help='Where to place generated images')
    parser.add_argument('--writers', nargs='+', default=None,
                        help='One or more writer IDs to process. If omitted, uses all writers present in target_file.')
    args = parser.parse_args()

    model_path = MODEL_DEFAULT.format(epoch=args.epoch)
    words_by_writer = load_words_per_writer(args.target_file)

    if not words_by_writer:
        print(f"❌ No (writer,word) pairs parsed from {args.target_file}.")
        sys.exit(1)

    if args.writers:
        target_wids = [w for w in args.writers if w in words_by_writer]
        missing = sorted(set(args.writers) - set(target_wids))
        if missing:
            print(f"⚠️ Requested writer(s) not found in {args.target_file}: {missing}")
    else:
        target_wids = sorted(words_by_writer.keys())

    # output folder per run (same structure you had)
    run_folder = os.path.join(args.out_base, str(args.epoch), 'res_by_writer_words')
    os.makedirs(run_folder, exist_ok=True)
    print(f'Output -> {run_folder}')

    for wid in tqdm(target_wids):
        out_folder = os.path.join(run_folder, wid)
        words = words_by_writer[wid]
        # de-duplicate but keep stable order
        seen = set(); filtered_words = []
        for w in words:
            if w not in seen:
                filtered_words.append(w); seen.add(w)
        generate_for_writer(wid, model_path, out_folder, filtered_words, args.img_base)

if __name__ == '__main__':
    main()
