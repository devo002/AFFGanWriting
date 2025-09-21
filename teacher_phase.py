# teacher_phase.py
import torch
from typing import Dict, Any

from helpers import (
    generate_from_batch,
    texts_to_labels,
    recognition_logits,
    recognition_loss,
)

@torch.no_grad()
def _freeze_except_rec(model):
    for p in model.gen.parameters(): p.requires_grad = False
    for p in model.dis.parameters(): p.requires_grad = False
    for p in model.cla.parameters(): p.requires_grad = False
    for p in model.rec.parameters(): p.requires_grad = True

def _restore_train_flags(model):
    for p in model.gen.parameters(): p.requires_grad = True
    for p in model.dis.parameters(): p.requires_grad = True
    for p in model.cla.parameters(): p.requires_grad = True

def run_teacher_phase_fake(
    model,
    loader,                    # typically test_loader or a dedicated guidance loader
    trocr_teacher,
    rec_lr: float = 1e-5,
    conf_threshold: float = 0.6,
    max_steps: int = 400,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """
    TrOCR on generated (synthetic) images -> train HTR only.
    Returns stats dict for logging.
    """
    device = next(model.parameters()).device

    _freeze_except_rec(model)
    model.gen.eval()  # BN-safe when batch size is small for generator

    rec_guidance_opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.rec.parameters()),
        lr=rec_lr
    )

    used_batches = used_samples = skipped_small = 0
    loss_sum = conf_sum = 0.0
    steps = 0

    for batch in loader:
        if steps >= max_steps:
            break

        # (a) generate fakes (no grad to G)
        with torch.no_grad():
            xg = generate_from_batch(model, batch).detach()  # [B,1,H,W] in [-1,1]

        # (b) TrOCR inference -> pseudo labels + confidence
        texts, conf = trocr_teacher.predict(xg)              # conf: [B]
        mask = conf >= conf_threshold
        n_used = int(mask.sum().item())
        if n_used < 2:
            skipped_small += 1
            continue

        # (c) labels
        labels = texts_to_labels([t for i, t in enumerate(texts) if mask[i]])
        xg_sel = xg[mask]

        # (d) recognizer update on pseudo labels
        rec_logits = recognition_logits(model, xg_sel, labels['ids'], labels['img_width'])
        loss_rec_pseudo = recognition_loss(rec_logits, labels)

        mean_conf = float(conf[mask].mean().item())
        w = float(max(0.6, min(1.0, mean_conf)))
        loss = w * loss_rec_pseudo

        rec_guidance_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.rec.parameters(), grad_clip)
        rec_guidance_opt.step()

        used_batches += 1
        used_samples += n_used
        loss_sum += float(loss.item())
        conf_sum += mean_conf
        steps += 1

    # stats
    stats = {
        "used_batches": used_batches,
        "used_samples": used_samples,
        "skipped_small_batches": skipped_small,
        "avg_pseudo_loss": (loss_sum / used_batches) if used_batches > 0 else 0.0,
        "avg_confidence": (conf_sum / used_batches) if used_batches > 0 else 0.0,
    }
    return stats


def run_teacher_phase_real(
    model,
    loader,                    # typically train_loader
    trocr_teacher,
    rec_lr: float = 1e-5,
    conf_threshold: float = 0.6,
    max_steps: int = 200,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """
    TrOCR on real IAM images -> train HTR only.
    Returns stats dict for logging.
    """
    device = next(model.parameters()).device

    # We assume gen remains eval() from the fake pass; real pass doesn't use gen.
    # Still freeze others and enable rec.
    _freeze_except_rec(model)

    rec_guidance_opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.rec.parameters()),
        lr=rec_lr
    )

    used_batches = used_samples = skipped_small = 0
    loss_sum = conf_sum = 0.0
    steps = 0

    for batch in loader:
        if steps >= max_steps:
            break

        # unpack batch (matches your sort_batch order)
        (tr_domain, tr_wid, tr_idx,
         tr_img, tr_img_width, tr_label,
         img_xt, label_xt, label_xt_swap) = batch

        tr_img = tr_img.to(device)              # [B,1,H,W] in [-1,1]
        img_width_real_cpu = tr_img_width.long()  # keep CPU (decoder uses .numpy())

        # (a) TrOCR on REAL images (no grad)
        with torch.no_grad():
            texts_real, conf_real = trocr_teacher.predict(tr_img)

        mask_real = conf_real >= conf_threshold
        n_used_real = int(mask_real.sum().item())
        if n_used_real < 2:
            skipped_small += 1
            continue

        # (b) select images & widths, build labels
        tr_img_sel = tr_img[mask_real]
        idx_real = mask_real.detach().cpu().nonzero(as_tuple=False).squeeze(1)
        img_width_sel_cpu = img_width_real_cpu.index_select(0, idx_real)
        labels_real = texts_to_labels([t for i, t in enumerate(texts_real) if mask_real[i]])

        # (c) recognizer update on pseudo labels (REAL widths)
        rec_logits_real = recognition_logits(model, tr_img_sel, labels_real['ids'], img_width_sel_cpu)
        loss_rec_pseudo_real = recognition_loss(rec_logits_real, labels_real)

        mean_conf_real = float(conf_real[mask_real].mean().item())
        w_real = float(max(0.6, min(1.0, mean_conf_real)))
        loss_real = w_real * loss_rec_pseudo_real

        rec_guidance_opt.zero_grad(set_to_none=True)
        loss_real.backward()
        torch.nn.utils.clip_grad_norm_(model.rec.parameters(), grad_clip)
        rec_guidance_opt.step()

        used_batches += 1
        used_samples += n_used_real
        loss_sum += float(loss_real.item())
        conf_sum += mean_conf_real
        steps += 1

    stats = {
        "used_batches": used_batches,
        "used_samples": used_samples,
        "skipped_small_batches": skipped_small,
        "avg_pseudo_loss": (loss_sum / used_batches) if used_batches > 0 else 0.0,
        "avg_confidence": (conf_sum / used_batches) if used_batches > 0 else 0.0,
    }
    return stats


def run_teacher_phase_all(
    model,
    train_loader,
    test_loader,
    trocr_teacher,
    writer,
    epoch: int,
    rec_lr: float = 1e-5,
    conf_threshold: float = 0.6,
    max_steps_fake: int = 400,
    max_steps_real: int = 200,
    grad_clip: float = 1.0,
):
    """
    Convenience wrapper: run FAKE then REAL teacher phases,
    log both, and restore model state.
    """
    # Freeze non-rec, put G in eval for the whole micro-phase
    _freeze_except_rec(model)
    model.gen.eval()

    # Fake (generated) images
    fake_stats = run_teacher_phase_fake(
        model=model,
        loader=test_loader,
        trocr_teacher=trocr_teacher,
        rec_lr=rec_lr,
        conf_threshold=conf_threshold,
        max_steps=max_steps_fake,
        grad_clip=grad_clip,
    )

    # Real images
    real_stats = run_teacher_phase_real(
        model=model,
        loader=train_loader,
        trocr_teacher=trocr_teacher,
        rec_lr=rec_lr,
        conf_threshold=conf_threshold,
        max_steps=max_steps_real,
        grad_clip=grad_clip,
    )

    # Restore generator to train() for normal epochs and grads flags on others
    model.gen.train()
    _restore_train_flags(model)

    # Log to TB
    if writer is not None:
        writer.add_scalars("teacher_phase", {
            "avg_pseudo_loss": fake_stats["avg_pseudo_loss"],
            "avg_confidence": fake_stats["avg_confidence"],
            "used_batches": fake_stats["used_batches"],
            "used_samples": fake_stats["used_samples"],
            "skipped_small_batches": fake_stats["skipped_small_batches"],
        }, epoch)

        writer.add_scalars("teacher_phase_real", {
            "avg_pseudo_loss": real_stats["avg_pseudo_loss"],
            "avg_confidence": real_stats["avg_confidence"],
            "used_batches": real_stats["used_batches"],
            "used_samples": real_stats["used_samples"],
            "skipped_small_batches": real_stats["skipped_small_batches"],
        }, epoch)

    return fake_stats, real_stats
