import torch
import os
import re
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from loss_tro import CER
from load_data import tokens, num_tokens, index2letter

def parse_filename(filename):
    """
    Extract ground truth and prediction from filename: 000-1.wordA-wordB.png
    """
    try:
        core = os.path.splitext(filename)[0]  # remove .png
        match = re.match(r'\d+-\d+\.(.+)-(.+)', core)
        if match:
            gt, pred = match.groups()
            return gt, pred
        else:
            print(f" Skipping unparsable file: {filename}")
            return None, None
    except:
        return None, None

def text_to_indices(text):
    """
    Convert string to token indices with GO and END tokens.
    """
    indices = [tokens['GO_TOKEN']]
    for c in text:
        try:
            idx = [k for k, v in index2letter.items() if v == c]
            if idx:
                indices.append(idx[0] + num_tokens)
        except:
            continue
    indices.append(tokens['END_TOKEN'])
    return indices

def compute_cer_from_folder(folder):
    filenames = [f for f in os.listdir(folder) if f.endswith('.png')]
    cer = CER()

    total_ed = 0
    total_len = 0

    print(f"📁 Evaluating folder: {folder}")
    for fname in tqdm(filenames):
        gt, pred = parse_filename(fname)
        if gt is None or pred is None:
            continue

        # Text-based CER
        ed = levenshtein_distance(pred, gt)
        length = len(gt) if len(gt) > 0 else 1
        sample_cer = (ed / length) * 100
        print(f"[{fname}] CER: {sample_cer:.2f}% | GT: {gt} | Pred: {pred}")

        total_ed += ed
        total_len += length

        # Tensor-based CER accumulation
        gt_idx = text_to_indices(gt)
        pred_idx = text_to_indices(pred)

        max_len = max(len(gt_idx), len(pred_idx))
        pad = tokens['PAD_TOKEN']

        gt_tensor = torch.full((1, max_len), pad, dtype=torch.long)
        pred_tensor = torch.full((1, max_len), pad, dtype=torch.long)

        gt_tensor[0, :len(gt_idx)] = torch.tensor(gt_idx)
        pred_tensor[0, :len(pred_idx)] = torch.tensor(pred_idx)

        cer.add(pred_tensor, gt_tensor)

    overall_cer = (total_ed / total_len) * 100
    cer_tensor_cer = cer.fin()

    print("\n Results:")
    print(f" Final CER (text-based Levenshtein): {overall_cer:.2f}%")
    print(f" Final CER (tensor-based, via CER class): {cer_tensor_cer:.2f}%")

if __name__ == '__main__':
    # UPDATE THIS PATH to your image folder
    image_folder = '/home/vault/iwi5/iwi5333h/test_single_writer.190_scenarios/30003000/res_4.oo_vocab_te_writer'
    compute_cer_from_folder(image_folder)
