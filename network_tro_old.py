import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import vocab_size, IMG_WIDTH, OUTPUT_MAX_LEN, index2letter, num_tokens, tokens
from modules_tro import GenModel_FC, DisModel, WriterClaModel, RecModel, write_image
from loss_tro import recon_criterion, crit, log_softmax
import numpy as np

w_dis = 1.
w_cla = 1.
w_l1  = 0.
w_rec = 1.

gpu = torch.device('cuda')

# ========================= DEBUG HELPERS ============================
from typing import List

def _tok2ch(i: int) -> str:
    """Map your vocab id -> readable symbol."""
    try:
        if isinstance(tokens, dict):
            if "PAD_TOKEN" in tokens and i == tokens["PAD_TOKEN"]:
                return "<PAD>"
            if "GO_TOKEN" in tokens and i == tokens["GO_TOKEN"]:
                return "<GO>"
            if "EOS_TOKEN" in tokens and i == tokens["EOS_TOKEN"]:
                return "<EOS>"
            if "SPACE_TOKEN" in tokens and i == tokens["SPACE_TOKEN"]:
                return " "
    except Exception:
        pass
    j = int(i) - int(num_tokens)
    if 0 <= j < len(index2letter):
        return str(index2letter[j])
    return f"#{int(i)}"

@torch.no_grad()
def debug_print_rec(name: str, logits: torch.Tensor, labels_shifted: torch.Tensor,
                    pad_id: int, max_steps: int = 12, topk: int = 5, sample_idx: int = 0):
    """
    logits: [B,T,V] raw logits in YOUR vocab
    labels_shifted: [B,T] label ids aligned with logits (e.g., label[:,1:][:,:T])
    Prints logits stats, per-token CE (valid only), and top-k per step.
    """
    B, T, V = logits.shape
    print(f"\n[DEBUG {name}] logits shape={tuple(logits.shape)} labels shape={tuple(labels_shifted.shape)}")
    mean = logits.mean().item(); std = logits.std().item()
    lmin = logits.min().item();   lmax = logits.max().item()
    print(f"[DEBUG {name}] logits stats: mean={mean:.4f} std={std:.4f} min={lmin:.4f} max={lmax:.4f}")

    flat_ce = F.cross_entropy(logits.reshape(-1, V), labels_shifted.reshape(-1),
                              ignore_index=pad_id, reduction="none")
    valid = labels_shifted.reshape(-1) != pad_id
    if valid.any():
        print(f"[DEBUG {name}] CE mean(valid)={flat_ce[valid].mean().item():.4f} "
              f"n_valid={int(valid.sum().item())}/{labels_shifted.numel()}")
    else:
        print(f"[DEBUG {name}] WARNING: no valid targets")

    b = min(sample_idx, B-1)
    Tprint = max(0, min(max_steps, T))
    probs = torch.softmax(logits[b, :Tprint], dim=-1)  # [Tprint,V]
    topv, topi = probs.topk(k=min(topk, V), dim=-1)    # [Tprint,topk]
    for t in range(Tprint):
        gt = int(labels_shifted[b, t].item())
        tops = ", ".join([f"{_tok2ch(int(topi[t,k]))}:{topv[t,k].item():.2f}" for k in range(topi.shape[1])])
        print(f"  t={t:02d}  GT={_tok2ch(gt)!r:<6}  ARGMAX={_tok2ch(int(topi[t,0]))!r:<6}  | {tops}")
# ===================================================================


class ConTranModel(nn.Module):
    def __init__(self, num_writers, show_iter_num, oov):
        super(ConTranModel, self).__init__()
        self.gen = GenModel_FC(OUTPUT_MAX_LEN).to(gpu)
        self.cla = WriterClaModel(num_writers).to(gpu)
        self.dis = DisModel().to(gpu)
        self.rec = RecModel(pretrain=False).to(gpu)
        self.iter_num = 0
        self.show_iter_num = show_iter_num
        self.oov = oov

    def forward(self, train_data_list, epoch, mode, cer_func=None):
        tr_domain, tr_wid, tr_idx, tr_img, tr_img_width, tr_label, img_xt, label_xt, label_xt_swap = train_data_list
        tr_wid = tr_wid.to(gpu)
        tr_img = tr_img.to(gpu)
        tr_img_width = tr_img_width.to(gpu)
        tr_label = tr_label.to(gpu)
        img_xt = img_xt.to(gpu)
        label_xt = label_xt.to(gpu)
        label_xt_swap = label_xt_swap.to(gpu)
        batch_size = tr_domain.shape[0]

        if mode == 'rec_update':
            tr_img_rec = tr_img[:, 0:1, :, :]     # [B,1,64,W]
            tr_img_rec = tr_img_rec.requires_grad_()
            tr_label_rec = tr_label[:, 0, :]      # [B,T]
            pred_xt_tr = self.rec(tr_img_rec, tr_label_rec,
                                  img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
            # --- DEBUG (rec update) ---
            try:
                pad_id = tokens['PAD_TOKEN'] if isinstance(tokens, dict) and 'PAD_TOKEN' in tokens else 0
                labels_dbg = tr_label_rec[:, 1:]                               # drop <GO>
                Tdbg = min(pred_xt_tr.size(1), labels_dbg.size(1))
                debug_print_rec("REC_UPDATE",
                                pred_xt_tr[:, :Tdbg, :].detach().cpu(),
                                labels_dbg[:, :Tdbg].detach().cpu(),
                                pad_id)
                decu = self.rec.decode(tr_img_rec[:1])
                print(f"[DEBUG decode] REC_UPDATE img[0]: {decu[0] if len(decu)>0 else '<empty>'}")
            except Exception as e:
                print("[DEBUG REC_UPDATE] debug print failed:", repr(e))
            # --------------------------

            tr_label_rec2 = tr_label_rec[:, 1:]   # remove <GO>
            l_rec_tr = crit(log_softmax(pred_xt_tr.reshape(-1, vocab_size)), tr_label_rec2.reshape(-1))
            cer_func.add(pred_xt_tr, tr_label_rec2)
            l_rec_tr.backward()
            return l_rec_tr

        elif mode =='cla_update':
            tr_img_rec = tr_img[:, 0:1, :, :]     # [B,1,64,W]
            tr_img_rec = tr_img_rec.requires_grad_()
            l_cla_tr = self.cla(tr_img_rec, tr_wid)
            l_cla_tr.backward()
            return l_cla_tr

        elif mode == 'gen_update':
            self.iter_num += 1
            # ---- dis loss ----
            f_xss = self.gen.enc_image(tr_img)         # list; last is [B,512,8,27]
            f_xs = f_xss[-1]
            f_xt, f_embed = self.gen.enc_text(label_xt, f_xs.shape)
            f_mix = self.gen.mix(f_xss, f_embed)
            xg = self.gen.decode(f_mix, f_xss, f_embed, f_xt)                 # [B,1,64,128]
            l_dis_ori = self.dis.calc_gen_loss(xg)

            f_xt_swap, f_embed_swap = self.gen.enc_text(label_xt_swap, f_xs.shape)
            f_mix_swap = self.gen.mix(f_xss, f_embed_swap)
            xg_swap = self.gen.decode(f_mix_swap, f_xss, f_embed_swap, f_xt_swap)
            l_dis_swap = self.dis.calc_gen_loss(xg_swap)
            l_dis = (l_dis_ori + l_dis_swap) / 2.

            # ---- writer classifier loss ----
            l_cla_ori = self.cla(xg, tr_wid)
            l_cla_swap = self.cla(xg_swap, tr_wid)
            l_cla = (l_cla_ori + l_cla_swap) / 2.

            # ---- l1 loss ----
            if self.oov:
                l_l1 = torch.tensor(0.).to(gpu)
            else:
                l_l1 = recon_criterion(xg, img_xt)

            # ---- rec loss ----
            cer_te, cer_te2 = cer_func
            pred_xt = self.rec(xg, label_xt, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
            pred_xt_swap = self.rec(xg_swap, label_xt_swap, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))

            # --- DEBUG (gen update) ---
            try:
                pad_id = tokens['PAD_TOKEN'] if isinstance(tokens, dict) and 'PAD_TOKEN' in tokens else 0
                # Your current loss drops <GO> only; for debugging we align logits length with labels-1.
                labels_ori_dbg = label_xt[:, 1:]
                labels_swp_dbg = label_xt_swap[:, 1:]
                Tori = min(pred_xt.size(1), labels_ori_dbg.size(1))
                Tswp = min(pred_xt_swap.size(1), labels_swp_dbg.size(1))

                debug_print_rec("GEN_ORI",
                                pred_xt[:, :Tori, :].detach().cpu(),
                                labels_ori_dbg[:, :Tori].detach().cpu(),
                                pad_id)
                debug_print_rec("GEN_SWP",
                                pred_xt_swap[:, :Tswp, :].detach().cpu(),
                                labels_swp_dbg[:, :Tswp].detach().cpu(),
                                pad_id)
                # Free-decode for qualitative read
                dec1 = self.rec.decode(xg[:1]);  dec2 = self.rec.decode(xg_swap[:1])
                print(f"[DEBUG decode] xg[0]     : {dec1[0] if len(dec1)>0 else '<empty>'}")
                print(f"[DEBUG decode] xg_swap[0]: {dec2[0] if len(dec2)>0 else '<empty>'}")
            except Exception as e:
                print("[DEBUG GEN_UPDATE] debug print failed:", repr(e))
            # -----------------------------

            label_xt2 = label_xt[:, 1:]                 # remove <GO>
            label_xt2_swap = label_xt_swap[:, 1:]       # remove <GO>
            l_rec_ori  = crit(log_softmax(pred_xt.reshape(-1, vocab_size)),       label_xt2.reshape(-1))
            l_rec_swap = crit(log_softmax(pred_xt_swap.reshape(-1, vocab_size)),  label_xt2_swap.reshape(-1))
            cer_te.add(pred_xt, label_xt2)
            cer_te2.add(pred_xt_swap, label_xt2_swap)
            l_rec = (l_rec_ori + l_rec_swap) / 2.

            # ---- total ----
            l_total = w_dis * l_dis + w_cla * l_cla + w_l1 * l_l1 + w_rec * l_rec
            l_total.backward()
            return l_total, l_dis, l_cla, l_l1, l_rec

        elif mode == 'dis_update':
            sample_img1 = tr_img[:,0:1,:,:]
            sample_img2 = tr_img[:,1:2,:,:]
            sample_img1.requires_grad_()
            sample_img2.requires_grad_()
            l_real1 = self.dis.calc_dis_real_loss(sample_img1)
            l_real2 = self.dis.calc_dis_real_loss(sample_img2)
            l_real = (l_real1 + l_real2) / 2.
            l_real.backward(retain_graph=True)

            with torch.no_grad():
                f_xss = self.gen.enc_image(tr_img)
                f_xs = f_xss[-1]
                f_xt, f_embed = self.gen.enc_text(label_xt, f_xs.shape)
                f_mix = self.gen.mix(f_xss, f_embed)
                xg = self.gen.decode(f_mix, f_xss, f_embed, f_xt)
                f_xt_swap, f_embed_swap = self.gen.enc_text(label_xt_swap, f_xs.shape)
                f_mix_swap = self.gen.mix(f_xss, f_embed_swap)
                xg_swap = self.gen.decode(f_mix_swap, f_xss, f_embed_swap, f_xt_swap)

            l_fake_ori = self.dis.calc_dis_fake_loss(xg)
            l_fake_swap = self.dis.calc_dis_fake_loss(xg_swap)
            l_fake = (l_fake_ori + l_fake_swap) / 2.
            l_fake.backward()

            l_total = l_real + l_fake
            # write sample images + (optionally) predictions
            if self.iter_num % self.show_iter_num == 0:
                with torch.no_grad():
                    pred_xt = self.rec(xg, label_xt, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
                    pred_xt_swap = self.rec(xg_swap, label_xt_swap, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
                write_image(xg, pred_xt, img_xt, label_xt, tr_img, xg_swap, pred_xt_swap, label_xt_swap,
                            'epoch_'+str(epoch)+'-'+str(self.iter_num))
            return l_total

        elif mode =='eval':
            with torch.no_grad():
                f_xss = self.gen.enc_image(tr_img)
                f_xs = f_xss[-1]
                f_xt, f_embed = self.gen.enc_text(label_xt, f_xs.shape)
                f_mix = self.gen.mix(f_xss, f_embed)
                xg = self.gen.decode(f_mix, f_xss, f_embed, f_xt)

                f_xt_swap, f_embed_swap = self.gen.enc_text(label_xt_swap, f_xs.shape)
                f_mix_swap = self.gen.mix(f_xss, f_embed_swap)
                xg_swap = self.gen.decode(f_mix_swap, f_xss, f_embed_swap, f_xt_swap)

                pred_xt = self.rec(xg, label_xt, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
                pred_xt_swap = self.rec(xg_swap, label_xt_swap, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
                write_image(xg, pred_xt, img_xt, label_xt, tr_img, xg_swap, pred_xt_swap, label_xt_swap,
                            'eval_'+str(epoch)+'-'+str(self.iter_num))
                self.iter_num += 1

                # ---- DEBUG (eval) ----
                try:
                    pad_id = tokens['PAD_TOKEN'] if isinstance(tokens, dict) and 'PAD_TOKEN' in tokens else 0
                    labels_ori_dbg = label_xt[:, 1:];  labels_swp_dbg = label_xt_swap[:, 1:]
                    Tori = min(pred_xt.size(1), labels_ori_dbg.size(1))
                    Tswp = min(pred_xt_swap.size(1), labels_swp_dbg.size(1))
                    debug_print_rec("EVAL_ORI", pred_xt[:, :Tori, :].cpu(), labels_ori_dbg[:, :Tori].cpu(), pad_id)
                    debug_print_rec("EVAL_SWP", pred_xt_swap[:, :Tswp, :].cpu(), labels_swp_dbg[:, :Tswp].cpu(), pad_id)
                    dec1 = self.rec.decode(xg[:1]);  dec2 = self.rec.decode(xg_swap[:1])
                    print(f"[DEBUG decode] eval xg[0]     : {dec1[0] if len(dec1)>0 else '<empty>'}")
                    print(f"[DEBUG decode] eval xg_swap[0]: {dec2[0] if len(dec2)>0 else '<empty>'}")
                except Exception as e:
                    print("[DEBUG EVAL] debug print failed:", repr(e))
                # ----------------------

                # ---- dis loss ----
                l_dis_ori = self.dis.calc_gen_loss(xg)
                l_dis_swap = self.dis.calc_gen_loss(xg_swap)
                l_dis = (l_dis_ori + l_dis_swap) / 2.

                # ---- rec loss (your original) ----
                cer_te, cer_te2 = cer_func
                label_xt2 = label_xt[:, 1:]
                label_xt2_swap = label_xt_swap[:, 1:]
                l_rec_ori  = crit(log_softmax(pred_xt.reshape(-1, vocab_size)),       label_xt2.reshape(-1))
                l_rec_swap = crit(log_softmax(pred_xt_swap.reshape(-1, vocab_size)),  label_xt2_swap.reshape(-1))
                cer_te.add(pred_xt, label_xt2)
                cer_te2.add(pred_xt_swap, label_xt2_swap)
                l_rec = (l_rec_ori + l_rec_swap) / 2.

                # ---- writer classifier loss ----
                l_cla_ori = self.cla(xg, tr_wid)
                l_cla_swap = self.cla(xg_swap, tr_wid)
                l_cla = (l_cla_ori + l_cla_swap) / 2.

            return l_dis, l_cla, l_rec
