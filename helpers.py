# helpers.py
import torch
import numpy as np
from loss_tro import crit, log_softmax
from load_data import index2letter, num_tokens, tokens, OUTPUT_MAX_LEN, IMG_WIDTH, vocab_size

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import Levenshtein as Lev
from load_data import letter2index  # same mapping used in writertest

@torch.no_grad()
def _labels_like_writertest(words):
    """
    Build labels exactly like writertest: [GO] + (letters+num_tokens) + [END] + PAD
    """
    B = len(words)
    T = OUTPUT_MAX_LEN
    ids = torch.full((B, T), tokens['PAD_TOKEN'], dtype=torch.long, device=gpu)

    for b, w in enumerate(words):
        ll = [letter2index[ch] for ch in w if ch in letter2index]
        ll = [t + num_tokens for t in ll]
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        if len(ll) < T:
            ll = ll + [tokens['PAD_TOKEN']] * (T - len(ll))
        else:
            ll = ll[:T]
        ids[b] = torch.tensor(ll, dtype=torch.long, device=gpu)

    img_width = torch.full((B,), IMG_WIDTH, dtype=torch.long)
    return ids, img_width

@torch.no_grad()
def generate_from_words_like_writertest(model, style_imgs, words,
                                        use_rec_filter=True, max_edit=100):
    assert len(words) == style_imgs.size(0)
    B = style_imgs.size(0)

    # Encode style ONCE for the full batch
    f_xss_full = model.gen.enc_image(style_imgs)  # list of pyramid feats, each [B,C,H,W]

    label_ids, img_widths = _labels_like_writertest(words)

    xg_list, keep_ids = [], []
    for i in range(B):
        # ----- slice per-sample features so batch dims match (1 vs 1) -----
        f_xss_i = [feat[i:i+1] for feat in f_xss_full]  # each [1,C,H,W]
        f_xs_i  = f_xss_i[-1]
        lid     = label_ids[i:i+1]                      # [1,T]

        # text encode for this sample
        f_xt, f_embed = model.gen.enc_text(lid, f_xs_i.shape)  # f_embed [1,C,H,W] (whatever your impl returns)

        # mix and decode for this sample
        f_mix = model.gen.mix(f_xss_i, f_embed)
        xg    = model.gen.decode(f_mix, f_xss_i, f_embed, f_xt)   # [1,1,H,W]

        if use_rec_filter:
            pred_logits = model.rec(xg, lid, img_width=img_widths[i:i+1])
            #top1 = torch.topk(pred_logits, 1, dim=-1)[1].squeeze(0).tolist()
            top1 = torch.topk(pred_logits, 1, dim=-1)[1].squeeze(0).squeeze(-1)  # [T]
            pred_ids = top1.tolist()
            pred_ids = [t for t in pred_ids if t >= num_tokens]
            gold_ids = [t for t in lid.squeeze(0).tolist() if t >= num_tokens]

            pred_txt = ''.join(index2letter[t - num_tokens] for t in pred_ids)
            gold_txt = ''.join(index2letter[t - num_tokens] for t in gold_ids)

            if Lev.distance(pred_txt, gold_txt) <= max_edit:
                xg_list.append(xg); keep_ids.append(i)
        else:
            xg_list.append(xg); keep_ids.append(i)

    if not xg_list:
        return None, [], None, None

    xg_stack         = torch.cat(xg_list, dim=0)         # [N,1,H,W]
    words_kept       = [words[i] for i in keep_ids]
    labels_ids_kept  = label_ids[keep_ids]               # [N,T]
    img_widths_kept  = img_widths[keep_ids]              # [N]
    return xg_stack, words_kept, labels_ids_kept, img_widths_kept

@torch.no_grad()
def trocr_predict_best_polarity(trocr_teacher, xg):
    """
    Try normal and inverted polarity; return the better one by mean confidence.
    xg: [-1,1] from generator
    """
    x01 = (xg + 1.0) / 2.0
    t1, c1 = trocr_teacher.predict(x01)
    t2, c2 = trocr_teacher.predict(1.0 - x01)
    return (t2, c2, 1.0 - x01) if c2.mean().item() > c1.mean().item() else (t1, c1, x01)


@torch.no_grad()
def generate_from_batch(model, train_data_list):
    """
    Generate fake images using the generator inside model.
    This mirrors 'gen_update' forward, but without gradients.
    Returns: xg [B,1,H,W] in [-1,1].
    """
    tr_domain, tr_wid, tr_idx, tr_img, tr_img_width, tr_label, img_xt, label_xt, label_xt_swap = train_data_list
    tr_img = tr_img.to(gpu)
    label_xt = label_xt.to(gpu)

    f_xss = model.gen.enc_image(tr_img)
    f_xs = f_xss[-1]
    f_xt, f_embed = model.gen.enc_text(label_xt, f_xs.shape)
    f_mix = model.gen.mix(f_xss, f_embed)
    xg = model.gen.decode(f_mix, f_xss, f_embed, f_xt)   # [B,1,H,W]
    return xg

@torch.no_grad()
def generate_from_words(model, style_imgs, words):
    """
    style_imgs: [B,1,H,W] style images to drive appearance (B must match len(words); repeat rows if needed)
    words: list[str] length B
    Returns: xg [B,1,H,W] in [-1,1]
    """
    from helpers import texts_to_labels  # if same file, not needed
    B = len(words)
    device = style_imgs.device
    labels = texts_to_labels(words)           # {"ids": [B,T], ...}
    label_ids = labels["ids"].to(device)

    # style encoding
    f_xss = model.gen.enc_image(style_imgs)   # list of pyramid feats
    f_xs = f_xss[-1]

    # text conditioning from desired words
    f_xt, f_embed = model.gen.enc_text(label_ids, f_xs.shape)

    f_mix = model.gen.mix(f_xss, f_embed)
    xg = model.gen.decode(f_mix, f_xss, f_embed, f_xt)
    return xg


# cached char maps to avoid rebuilding each call
_char2id = None
_pad = None
_go = None
_eos = None

def _build_char_maps():
    global _char2id, _pad, _go, _eos
    if _char2id is None:
        _char2id = {ch: (num_tokens + i) for i, ch in enumerate(index2letter)}
        _pad = tokens['PAD_TOKEN']
        _go  = tokens.get('GO_TOKEN', None)
        _eos = tokens.get('END_TOKEN', None)


def texts_to_labels(texts):
    """
    Convert list[str] into your label tensor format.
    Returns dict: ids [B,T], img_width [B], lengths [B]
    """
    _build_char_maps()
    B = len(texts)
    T = OUTPUT_MAX_LEN
    ids = torch.full((B, T), _pad, dtype=torch.long, device=gpu)

    # optional <GO> at pos 0
    start = 0
    if _go is not None:
        ids[:, 0] = _go
        start = 1

    for b, txt in enumerate(texts):
        pos = start
        for ch in txt:
            if pos >= T: break
            if ch in _char2id:
                ids[b, pos] = _char2id[ch]
                pos += 1
        if (pos < T) and (_eos is not None):
            ids[b, pos] = _eos

    img_width = torch.from_numpy(np.array([IMG_WIDTH]*B))
    lengths = torch.tensor([T]*B)
    return {"ids": ids, "img_width": img_width, "lengths": lengths}


def recognition_logits(model, img, label_ids, img_width):
    """
    Forward through recognizer -> logits [B,T,V].
    """
    return model.rec(img, label_ids, img_width)


def recognition_loss(logits, labels):
    """
    Recognition CE/LabelSmoothing loss.
    Drops <GO> before comparing.
    """
    label_ids = labels["ids"]
    target = label_ids[:, 1:] if label_ids.size(1) > 1 else label_ids
    return crit(log_softmax(logits.reshape(-1, vocab_size)), target.reshape(-1))


# Put this near the top of helpers.py (outside any function)
#TARGET_WORDS = [
#    "enigma", "egghead", "gripped", "Tuesday", "possess",
#    "returns", "critics", "Privy", "massive", "blonde", "accents",
#]

# # helpers.py
TARGET_WORDS = [
    "Members","of","the","Cabinet","are","basing","their","on","new",
    "booklet","called","The","Record","Speaks","which","in","some","detail",
    "the","of","the","party","since","it","came","to","office","three","and",
    "half","years","ago","there","is","little","in","the","Party","that",
    "their","stock","at","home","has","fallen","in","the","face","of","heavy",
    "and","an","economy","The","process","has","been","too","slow","for",
    "Herr","Strauss","and","last","month","he","Britain","for","being","an",
    "for","West","plans","for"
]

#TARGET_WORDS2 = [
#    "warrior","Morning","poetic","nodding","certify","reviews","mosaics","senders","humming","bumped","redeem","robbing","Married","robbing","witches","visibly","arsenal","skiing","windy","recieve","flower","voyager","noisy","moody","Isle"
#    ,"rackets","cloud","thunder","Glow","handle","grades","Clever","parties","friends"
#]
#]
#]
#]
#]
