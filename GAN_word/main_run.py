# main_runnew.py  — FULL UPDATED FILE

import os
import glob
import time
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn
import optuna

from load_data import NUM_WRITERS
from load_data import loadData as load_data_func
from network_tro import ConTranModel
from loss_tro import CER
from torch.optim.lr_scheduler import ReduceLROnPlateau

# NEW: TrOCR teacher + helpers
from trocr_teacher import TrocrTeacher
from helpers import (
    generate_from_batch,
    texts_to_labels,
    recognition_logits,
    recognition_loss,
    generate_from_words,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter


# ---------------- Args ----------------
parser = argparse.ArgumentParser(
    description='AFFGANwriting main runner',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
parser.add_argument('--optuna', action='store_true', help='Run Optuna hyperparameter tuning')
args = parser.parse_args()

gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

# ---------------- Config ----------------
OOV = True
NUM_THREAD = 2
EARLY_STOP_EPOCH = 20
EVAL_EPOCH = 20
MODEL_SAVE_EPOCH = 100
show_iter_num = 500

BATCH_SIZE = 8
lr_dis = 1e-4
lr_gen = 1e-4
lr_rec = 1e-5
lr_cla = 1e-5

CurriculumModelID = args.start_epoch
model_name = 'aaa'
run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
base_logdir = '/home/woody/iwi5/iwi5333h'
logdir = os.path.join(base_logdir, 'log11', model_name + '-' + str(run_id))
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)


def log(msg):
    logger = logging.getLogger("GanWriting")
    logger.setLevel(logging.INFO)
    log_path = os.path.join(logdir, 'log.txt')
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.info(msg)
    logger.removeHandler(handler)
    logger.removeHandler(console)


# ---------------- Data ----------------
def sort_batch(batch):
    train_domain, train_wid, train_idx = [], [], []
    train_img, train_img_width, train_label = [], [], []
    img_xts, label_xts, label_xts_swap = [], [], []

    for domain, wid, idx, img, img_width, label, img_xt, label_xt, label_xt_swap in batch:
        if wid >= NUM_WRITERS:
            print('error!')
        train_domain.append(domain)
        train_wid.append(wid)
        train_idx.append(idx)
        train_img.append(img)
        train_img_width.append(img_width)
        train_label.append(label)
        img_xts.append(img_xt)
        label_xts.append(label_xt)
        label_xts_swap.append(label_xt_swap)

    return (
        np.array(train_domain),
        torch.tensor(train_wid, dtype=torch.int64),
        np.array(train_idx),
        torch.tensor(np.array(train_img), dtype=torch.float32),
        torch.tensor(np.array(train_img_width), dtype=torch.int64),
        torch.tensor(np.array(train_label), dtype=torch.int64),
        torch.tensor(np.array(img_xts), dtype=torch.float32),
        torch.tensor(np.array(label_xts), dtype=torch.int64),
        torch.tensor(np.array(label_xts_swap), dtype=torch.int64),
    )


def all_data_loader():
    data_train, data_test = load_data_func(OOV)
    train_loader = torch.utils.data.DataLoader(
        data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_THREAD, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_THREAD, pin_memory=True
    )
    return train_loader, test_loader


# ---------------- Train / Eval ----------------
def train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch):
    model.train()
    loss_dis, loss_dis_tr = [], []
    loss_cla, loss_cla_tr = [], []
    loss_l1, loss_rec, loss_rec_tr = [], [], []

    time_s = time.time()
    cer_tr = CER()
    cer_te = CER()
    cer_te2 = CER()

    for train_data_list in train_loader:
        # recognizer (HTR) update on real imgs
        rec_opt.zero_grad()
        l_rec_tr = model(train_data_list, epoch, 'rec_update', cer_tr)
        rec_opt.step()

        # classifier update
        cla_opt.zero_grad()
        l_cla_tr = model(train_data_list, epoch, 'cla_update')
        cla_opt.step()

        # discriminator update
        dis_opt.zero_grad()
        l_dis_tr = model(train_data_list, epoch, 'dis_update')
        dis_opt.step()

        # generator update (uses recognizer on generated imgs)
        gen_opt.zero_grad()
        l_total, l_dis, l_cla, l_l1, l_rec = model(
            train_data_list, epoch, 'gen_update', [cer_te, cer_te2]
        )
        gen_opt.step()

        loss_dis.append(l_dis.detach().item())
        loss_dis_tr.append(l_dis_tr.detach().item())
        loss_cla.append(l_cla.detach().item())
        loss_cla_tr.append(l_cla_tr.detach().item())
        loss_l1.append(l_l1.detach().item())
        loss_rec.append(l_rec.detach().item())
        loss_rec_tr.append(l_rec_tr.detach().item())

    res_cer_tr = cer_tr.fin()
    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()

    writer.add_scalars("train", {
        "fl_dis_tr": np.mean(loss_dis_tr), "fl_dis": np.mean(loss_dis),
        "fl_cla_tr": np.mean(loss_cla_tr), "fl_cla": np.mean(loss_cla),
        "fl_rec_tr": np.mean(loss_rec_tr), "fl_rec": np.mean(loss_rec),
        "fl_l1": np.mean(loss_l1),
        "res_cer_tr": res_cer_tr, "res_cer_te": res_cer_te, "res_cer_te2": res_cer_te2
    }, epoch)

    log(('epo%d <tr>-<gen>: l_dis=%.2f-%.2f, l_cla=%.2f-%.2f, l_rec=%.2f-%.2f, l1=%.2f, cer=%.2f-%.2f-%.2f, time=%.1f' %
         (epoch, np.mean(loss_dis_tr), np.mean(loss_dis), np.mean(loss_cla_tr), np.mean(loss_cla),
          np.mean(loss_rec_tr), np.mean(loss_rec), np.mean(loss_l1),
          res_cer_tr, res_cer_te, res_cer_te2, time.time() - time_s)))

    return res_cer_te + res_cer_te2


def test(test_loader, epoch, modelFile_o_model):
    if isinstance(modelFile_o_model, str):
        model = ConTranModel(NUM_WRITERS, show_iter_num, OOV).to(gpu)
        print('Loading ' + modelFile_o_model)
        model.load_state_dict(torch.load(modelFile_o_model))
    else:
        model = modelFile_o_model
    model.eval()

    loss_dis, loss_cla, loss_rec = [], [], []
    time_s = time.time()
    cer_te, cer_te2 = CER(), CER()

    for test_data_list in test_loader:
        l_dis, l_cla, l_rec = model(test_data_list, epoch, 'eval', [cer_te, cer_te2])
        loss_dis.append(l_dis.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_rec.append(l_rec.cpu().item())

    test_cer = cer_te.fin() + cer_te2.fin()

    writer.add_scalars("EVAL", {
        "fl_dis": np.mean(loss_dis), "fl_cla": np.mean(loss_cla),
        "fl_rec": np.mean(loss_rec),
        "res_cer_te": cer_te.fin(), "res_cer_te2": cer_te2.fin()
    }, epoch)

    log(('EVAL: l_dis=%.3f, l_cla=%.3f, l_rec=%.3f, cer=%.2f-%.2f, time=%.1f' %
         (np.mean(loss_dis), np.mean(loss_cla), np.mean(loss_rec),
          cer_te.fin(), cer_te2.fin(), time.time() - time_s)))

    return test_cer


# ---------------- Early Stopping ----------------
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss


# ---------------- Main loop ----------------
def main(train_loader, test_loader, num_writers):
    model = ConTranModel(num_writers, show_iter_num, OOV).to(gpu)
    folder_weights = '/home/vault/iwi5/iwi5333h/save_weights11'
    os.makedirs(folder_weights, exist_ok=True)

    if CurriculumModelID > 0:
        model_file = os.path.join(folder_weights, f'contran-{CurriculumModelID}.model')
        print('Loading ' + model_file)
        model.load_state_dict(torch.load(model_file))

    dis_opt = optim.Adam(filter(lambda p: p.requires_grad, model.dis.parameters()), lr=lr_dis)
    gen_opt = optim.Adam(filter(lambda p: p.requires_grad, model.gen.parameters()), lr=lr_gen)
    rec_opt = optim.Adam(filter(lambda p: p.requires_grad, model.rec.parameters()), lr=lr_rec)
    cla_opt = optim.Adam(filter(lambda p: p.requires_grad, model.cla.parameters()), lr=lr_cla)
    sched_rec = ReduceLROnPlateau(rec_opt, mode='min', factor=0.5, patience=5)


    # NEW: frozen TrOCR teacher (official HF). Use hub ID or a local path.
    trocr_teacher = TrocrTeacher("/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten", device=gpu)

    early_stopping = EarlyStopping(
        patience=EARLY_STOP_EPOCH if EARLY_STOP_EPOCH else 20, verbose=True
    )

    for epoch in range(CurriculumModelID, 50001):
        if epoch > 4000:
            global MODEL_SAVE_EPOCH
            MODEL_SAVE_EPOCH = 20

        if epoch % 20 == 0 and epoch != 0:
            train_loader, test_loader = all_data_loader()

        _ = train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch)

        if epoch % EVAL_EPOCH == 0:
            # 1) normal validation (metrics first, no grads)
            test_cer = test(test_loader, epoch, model)
            early_stopping(test_cer)
            sched_rec.step(test_cer)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                model_path = os.path.join(folder_weights, f'contran-best.model')
                torch.save(model.state_dict(), model_path)
                break
            
            if epoch >= 800:
            # 2) ---- TrOCR-guided HTR teacher phase (R-only) ----
                model.train()

            # freeze everything except recognizer
                for p in model.gen.parameters(): p.requires_grad = False
                for p in model.dis.parameters(): p.requires_grad = False
                for p in model.cla.parameters(): p.requires_grad = False
                for p in model.rec.parameters(): p.requires_grad = True

            # generator eval() to avoid BN crash on tiny batches
                model.gen.eval()

                rec_guidance_opt = optim.Adam(
                filter(lambda p: p.requires_grad, model.rec.parameters()),
                lr=1e-5
                )
                steps, max_steps = 0, 200
                conf_threshold = 0.6
                grad_clip = 1.0
            
                used_batches = 0
                used_samples = 0
                skipped_small = 0
                loss_sum = 0.0
                conf_sum = 0.0

                for batch in test_loader:  # or a dedicated guidance_loader
                    if steps >= max_steps:
                        break

                # (a) generate fakes with current G (no grad)
                    with torch.no_grad():
                        xg = generate_from_batch(model, batch).detach()

                # (b) TrOCR inference -> pseudo labels + confidence
                    texts, conf = trocr_teacher.predict(xg)
                    mask = conf >= conf_threshold

                # Skip too-small micro-batches so BN in recognizer doesn't error
                    n_used = int(mask.sum().item())
                    if n_used < 2:
                        skipped_small += 1
                        continue
                

                # (c) texts -> label ids (CPU img_width handled in helper)
                    labels = texts_to_labels([t for i, t in enumerate(texts) if mask[i]])
                    xg_sel = xg[mask]

                # (d) recognizer forward + CE on pseudo labels
                    rec_logits = recognition_logits(model, xg_sel, labels['ids'], labels['img_width'])
                    loss_rec_pseudo = recognition_loss(rec_logits, labels)

                # optional: average-confidence weighting
                    mean_conf = float(conf[mask].mean().item())
                    w = float(max(0.6, min(1.0, mean_conf)))
                    loss = w * loss_rec_pseudo
                
                #w = float(conf[mask].mean().clamp(0.6, 1.0).item())
                #loss = w * loss_rec_pseudo

                    rec_guidance_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.rec.parameters(), grad_clip)
                    rec_guidance_opt.step()
                
                    used_batches += 1
                    used_samples += n_used
                    loss_sum += float(loss.item())
                    conf_sum += mean_conf

                    steps += 1

            # restore generator to train() for next epoch's standard training
                model.gen.train()

            # re-enable grads on others
                for p in model.gen.parameters(): p.requires_grad = True
                for p in model.dis.parameters(): p.requires_grad = True
                for p in model.cla.parameters(): p.requires_grad = True
            
                if used_batches > 0:
                    teacher_avg_loss = loss_sum / used_batches
                    teacher_avg_conf = conf_sum / used_batches
                else:
                    teacher_avg_loss = 0.0
                    teacher_avg_conf = 0.0

                writer.add_scalars("teacher_phase", {
                "avg_pseudo_loss": teacher_avg_loss,
                "avg_confidence": teacher_avg_conf,
                "used_batches": used_batches,
                "used_samples": used_samples,
                "skipped_small_batches": skipped_small
                }, epoch)

                log(f"[Teacher Phase @epoch {epoch}] "
                f"used_batches={used_batches}, used_samples={used_samples}, "
                f"avg_conf={teacher_avg_conf:.3f}, avg_pseudo_loss={teacher_avg_loss:.4f}, "
                f"skipped_small_batches={skipped_small}")
            

            if epoch % MODEL_SAVE_EPOCH == 0:
                model_path = os.path.join(folder_weights, f'contran-{epoch}.model')
                torch.save(model.state_dict(), model_path)


# ---------- Optuna helpers (unchanged core logic) ----------
def train_with_custom_lr(train_loader, test_loader, num_writers, lr_dis, lr_gen, lr_rec, lr_cla):
    model = ConTranModel(num_writers, show_iter_num, OOV).to(gpu)

    dis_opt = optim.Adam(filter(lambda p: p.requires_grad, model.dis.parameters()), lr=lr_dis)
    gen_opt = optim.Adam(filter(lambda p: p.requires_grad, model.gen.parameters()), lr=lr_gen)
    rec_opt = optim.Adam(filter(lambda p: p.requires_grad, model.rec.parameters()), lr=lr_rec)
    cla_opt = optim.Adam(filter(lambda p: p.requires_grad, model.cla.parameters()), lr=lr_cla)

    for epoch in range(160):
        train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch)
        if epoch % 20 == 0 and epoch != 0:
            val_cer = test(test_loader, epoch, model)
            print(f"[Trial Eval @Epoch {epoch}] CER: {val_cer:.2f}")

    final_cer = test(test_loader, 160, model)
    return final_cer


def optuna_objective(trial):
    lr_gen = trial.suggest_loguniform("lr_gen", 1e-5, 1e-3)
    lr_rec = trial.suggest_loguniform("lr_rec", 1e-6, 1e-4)
    lr_dis = 1e-4
    lr_cla = 1e-5
    train_loader, test_loader = all_data_loader()
    final_cer = train_with_custom_lr(train_loader, test_loader, NUM_WRITERS, lr_dis, lr_gen, lr_rec, lr_cla)
    return final_cer


def rm_old_model(index, folder_weights):
    models = glob.glob(os.path.join(folder_weights, '*.model'))
    for m in models:
        epoch = int(m.split('-')[-1].split('.')[0])
        if epoch < index:
            os.remove(m)


if __name__ == '__main__':
    print(time.ctime())
    train_loader, test_loader = all_data_loader()

    if args.optuna:
        storage_path = "sqlite:///optuna_effnetv2llr_study.db"
        study = optuna.create_study(
            study_name="gen_rec_lr_tuning",
            storage=storage_path,
            direction="minimize",
            load_if_exists=True
        )
        study.optimize(optuna_objective, n_trials=20)
        print("Best trial:")
        print(f"  CER: {study.best_trial.value}")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        main(train_loader, test_loader, NUM_WRITERS)

    print(time.ctime())
