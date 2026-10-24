"""Adım 8–9: küçük CNN autoencoder eğitimi (makine kimliği başına ayrı model).

Kullanım:
    python scripts/train_autoencoder.py --overfit-test --machine-ids id_00   # önce akıl sağlığı testi
    python scripts/train_autoencoder.py --machine-ids id_00                  # tek kimlik prototipi
    python scripts/train_autoencoder.py                                      # tüm kimlikler
    python scripts/train_autoencoder.py --seed 43                            # tohum tekrarı

Çıktılar (checkpoints/cnn_ae/seed<S>/<tür>/<id>/):
    best.pt, scaler.npz, meta.json, history.json, loss_curve.png, config.yaml (kopya)
"""
import _bootstrap  # noqa: F401

import argparse
import copy
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_split, machine_keys, subset
from src.detectors import detector_dir
from src.features import BandStandardizer, load_logmels, make_windows
from src.models import ConvAutoencoder, count_parameters
from src.utils import (append_run_log, ensure_dir, feature_settings_str, load_config, make_run_id,
                       resolve_device, save_json, set_seed)


def windows_of(logmels, scaler, fcfg):
    per_rec = [make_windows(scaler.transform(m), fcfg["frames"], fcfg["window_hop"])[0] for m in logmels]
    return per_rec


@torch.no_grad()
def record_level_loss(model, per_rec_windows, device, batch=256):
    """Pencere hatalarını önce kayıt içinde, sonra kayıtlar arasında ortalar."""
    model.eval()
    flat = np.concatenate(per_rec_windows)
    errs = []
    for i in range(0, len(flat), batch):
        x = torch.from_numpy(flat[i:i + batch][:, None]).to(device)
        errs.append(((model(x) - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
    errs = np.concatenate(errs)
    rec, k = [], 0
    for w in per_rec_windows:
        rec.append(errs[k:k + len(w)].mean())
        k += len(w)
    return float(np.mean(rec))


def overfit_test(cfg, logmels, device, steps=1000):
    """20 normal kayda aşırı öğrenme: kayıp belirgin düşmüyorsa boyut/dönüşüm/öğrenme oranında hata var.

    Fan sesi gürültü ağırlıklı olduğu için kayıp sıfıra inmez; ~1000 adımda en az %40 düşüş beklenir.
    """
    fcfg, acfg = cfg["features"], cfg["autoencoder"]
    scaler = BandStandardizer().fit(logmels, fcfg["std_eps"])
    X = torch.from_numpy(np.concatenate(windows_of(logmels, scaler, fcfg))[:, None]).to(device)
    model = ConvAutoencoder(fcfg["n_mels"], fcfg["frames"], acfg["bottleneck"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=acfg["lr"])

    def full_loss():
        with torch.no_grad():
            return float(((model(X) - X) ** 2).mean())

    first = full_loss()
    for step in range(steps):
        idx = torch.randint(0, len(X), (min(acfg["batch_size"], len(X)),), device=device)
        x = X[idx]
        out = model(x)
        assert out.shape == x.shape, f"Boyut uyuşmazlığı: {tuple(out.shape)} != {tuple(x.shape)}"
        loss = ((out - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 250 == 0:
            print(f"    adım {step + 1}: kayıp {full_loss():.4f}")
    last = full_loss()
    ok = last < 0.6 * first
    print(f"  Aşırı öğrenme testi: {len(logmels)} kayıt, {len(X)} pencere | kayıp {first:.4f} → {last:.4f} "
          f"| {'GEÇTİ' if ok else 'KALDI — boyut, dönüşüm ve öğrenme oranını kontrol edin'}")
    return ok


def train_one(cfg, split_df, mt, mid, seed, device):
    fcfg, acfg = cfg["features"], cfg["autoencoder"]
    set_seed(seed)
    t0 = time.time()
    train_lm = load_logmels(cfg, subset(split_df, mt, mid, "train"))
    val_lm = load_logmels(cfg, subset(split_df, mt, mid, "val"))
    scaler = BandStandardizer().fit(train_lm, fcfg["std_eps"])  # yalnızca eğitimden
    train_w = np.concatenate(windows_of(train_lm, scaler, fcfg))[:, None]
    val_w = windows_of(val_lm, scaler, fcfg)

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_w)), batch_size=acfg["batch_size"],
                        shuffle=True, generator=g, num_workers=acfg["num_workers"], drop_last=False)
    model = ConvAutoencoder(fcfg["n_mels"], fcfg["frames"], acfg["bottleneck"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=acfg["lr"])
    print(f"[cnn_ae] {mt} {mid} seed={seed}: {len(train_lm)} eğitim kaydı ({len(train_w)} pencere), "
          f"{len(val_lm)} doğrulama kaydı, {count_parameters(model):,} parametre, cihaz={device}")

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best, best_epoch, best_state, wait = float("inf"), -1, None, 0
    for epoch in range(1, acfg["max_epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for (x,) in loader:
            x = x.to(device)
            out = model(x)
            if epoch == 1 and n == 0:
                assert out.shape == x.shape, f"Boyut uyuşmazlığı: {tuple(out.shape)} != {tuple(x.shape)}"
            loss = ((out - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        tr = tot / n
        vl = record_level_loss(model, val_w, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        mark = ""
        if vl < best - 1e-6:
            best, best_epoch, best_state, wait = vl, epoch, copy.deepcopy(model.state_dict()), 0
            mark = " *"
        else:
            wait += 1
        print(f"  epoch {epoch:3d}  train {tr:.4f}  val {vl:.4f}{mark}")
        if wait >= acfg["patience"]:
            print(f"  Erken durdurma: {acfg['patience']} epoch iyileşme yok.")
            break
    dt = time.time() - t0

    out = ensure_dir(detector_dir(cfg, "cnn_ae", mt, mid, seed))
    torch.save(best_state, out / "best.pt")
    scaler.save(out / "scaler.npz")
    shutil.copy(cfg["_config_path"], out / "config.yaml")
    save_json(history, out / "history.json")
    meta = {"method": "cnn_ae", "machine_type": mt, "machine_id": mid, "seed": seed,
            "features": fcfg, "bottleneck": acfg["bottleneck"], "aggregation": cfg["scoring"]["aggregation"],
            "best_epoch": best_epoch, "best_val_loss": best, "epochs_run": len(history["epoch"]),
            "train_time_s": dt, "n_train_records": len(train_lm), "n_train_windows": int(len(train_w)),
            "n_params": count_parameters(model), "device": str(device)}
    save_json(meta, out / "meta.json")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history["epoch"], history["train_loss"], label="eğitim (pencere MSE)")
    ax.plot(history["epoch"], history["val_loss"], label="doğrulama (kayıt ort. MSE)")
    ax.axvline(best_epoch, ls="--", c="gray", lw=1, label=f"en iyi epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"CNN AE — {mt} {mid} (seed {seed})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=110)
    plt.close(fig)

    append_run_log(cfg, {
        "run_id": make_run_id("cnn_ae", mt, mid, seed), "method": "cnn_ae", "machine_type": mt,
        "machine_id": mid, "seed": seed, "feature_settings": feature_settings_str(cfg),
        "model": f"ConvAE(bottleneck={acfg['bottleneck']},lr={acfg['lr']},bs={acfg['batch_size']})",
        "train_time_s": f"{dt:.1f}", "best_epoch": best_epoch, "val_loss": f"{best:.5f}",
    })
    print(f"  En iyi epoch {best_epoch}, doğrulama {best:.4f}, süre {dt:.1f} s → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--machine-ids", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--overfit-test", action="store_true", help="Yalnızca 20 kayıtlık aşırı öğrenme testini çalıştır")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_epochs:
        cfg["autoencoder"]["max_epochs"] = args.max_epochs
    seed = cfg["seed"] if args.seed is None else args.seed
    device = resolve_device(cfg["autoencoder"]["device"])
    split_df = load_split(cfg)

    for mt, mid in machine_keys(split_df):
        if args.machine_ids and mid not in args.machine_ids:
            continue
        if args.overfit_test:
            set_seed(seed)
            lm = load_logmels(cfg, subset(split_df, mt, mid, "train").head(20))
            print(f"[overfit] {mt} {mid}")
            overfit_test(cfg, lm, device)
        else:
            train_one(cfg, split_df, mt, mid, seed, device)


if __name__ == "__main__":
    main()
