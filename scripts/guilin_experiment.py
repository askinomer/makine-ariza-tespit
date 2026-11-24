"""Guilin çok koşullu akustik rulman veri seti: CNN autoencoder ile Isolation Forest karşılaştırması.

    python scripts/guilin_experiment.py --wav /content/guilin_wav --out /content/guilin_results \
        --ckpt /content/guilin_ckpt --seeds 42 43 44

Girdi: tools/guilin_convert.py çıktısı (10 sn'lik 32.768 Hz WAV klipleri + manifest_<etiket>.csv).

PROTOKOL — koşul-dışı bırakma (leave-one-condition-out, 9 kat). Koşul = hız × yük.
  Her kat h için yalnızca NORMAL kayıtlar ve yalnızca h DIŞINDAKİ 8 koşul kullanılır:
    eğitim      : parça 1-3 (10 dk'lık dosyalar 1-3)    → model ve ölçekleyici
    doğrulama   : parça 4                                → erken durdurma (etiketsiz)
    kalibrasyon : parça 5                                → eşik = skorların 95. persentili
  Test: h koşulunun TÜM normal klipleri (hiç görülmemiş koşul/oturum) ve h'nin 4 arıza türü.
  Parçalar zaman bloklarıdır; komşu klipler aynı bölüme düşer (sızıntı yok). Arızalı kayıt eğitimde,
  model seçiminde ve eşikte hiç kullanılmaz.
Yöntemler: Isolation Forest (log mel ortalama+std, tohum 42) ve CNN AE (64×64 log mel pencereleri,
  kayıt skoru = pencere MSE ortalaması), CNN AE için 3 tohum.
Gürültü: test ve kalibrasyon kliplerine dosya başına sabit beyaz gürültü (+20, +10 dB SNR); eşik sabit ve
  gürültülü kalibrasyonla yeniden ayarlanmış iki satır.
Ek: aynı gün alt kümesi — 02.11.2025'te kaydedilmiş normal klipler ile dış bilezik (tümü 02.11).
Son olarak demo için tüm 9 koşulun normaliyle (aynı parça düzeni) birer son model eğitilir.
"""
import _bootstrap  # noqa: F401

import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.features import BandStandardizer, load_audio, logmel_from_waveform, make_windows
from src.metrics import threshold_metrics
from src.models import ConvAutoencoder, count_parameters
from src.noise import add_white_noise, file_rng
from src.utils import save_json, set_seed

FCFG = dict(sr=32768, n_fft=2048, win_length=2048, hop_length=1024, window="hann", n_mels=64, power=2.0,
            fmin=0, fmax=16384, top_db=80.0, ref=1.0, frames=64, window_hop=64, std_eps=1e-6)
ACFG = dict(bottleneck=128, lr=1e-3, batch_size=64, max_epochs=30, patience=5)
FAULTS = ["outer", "inner", "cage", "compound"]
FAULT_TR = {"outer": "dış bilezik", "inner": "iç bilezik", "cage": "kafes kırığı", "compound": "bileşik iç-dış",
            "tümü": "tüm arızalar"}
METHOD_TR = {"iforest": "Isolation Forest (referans)", "cnn_ae": "CNN autoencoder"}
NOISES = ["temiz", "20", "10"]
NOISE_SEED = 1234
PERCENTILE = 95
SAME_DAY = "2025-11-02"


# ---------------------------------------------------------------- veri ve önbellek
def load_table(wav: Path) -> pd.DataFrame:
    m = pd.concat([pd.read_csv(f) for f in sorted(wav.glob("manifest_*.csv"))], ignore_index=True)
    m = m.sort_values("path").reset_index(drop=True)
    m["cond"] = m["rpm"].astype(str) + "r_" + m["load"].astype(str)
    m["day"] = m["start_utc"].astype(str).str[:10]
    return m


def _logmel(path: str, snr):
    y = load_audio(path, FCFG["sr"])
    if snr != "temiz":
        y = add_white_noise(y, float(snr), file_rng(NOISE_SEED, float(snr), Path(path).name))
    return logmel_from_waveform(y, FCFG)


def logmel_cache(tab: pd.DataFrame, wav: Path, out: Path, snr: str) -> np.ndarray:
    f = out / "cache" / f"logmel_{snr}.npy"
    done = f.with_suffix(".done")
    if done.exists():
        return np.load(f, mmap_mode="r")
    f.parent.mkdir(parents=True, exist_ok=True)
    first = _logmel(str(wav / tab.path[0]), snr)
    arr = np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=(len(tab), *first.shape))
    t0, step = time.time(), 500
    for i in range(0, len(tab), step):
        res = Parallel(n_jobs=-1)(delayed(_logmel)(str(wav / p), snr) for p in tab.path[i:i + step])
        arr[i:i + len(res)] = np.stack(res)
        print(f"  log mel [{snr}] {min(i + step, len(tab))}/{len(tab)} · {time.time() - t0:.0f} s", flush=True)
    arr.flush()
    done.touch()
    return np.load(f, mmap_mode="r")


def baseline_vectors(L: np.ndarray) -> np.ndarray:
    return np.concatenate([L.mean(axis=2), L.std(axis=2)], axis=1).astype(np.float32)


# ---------------------------------------------------------------- CNN AE
def windows(L, idx, scaler):
    return [make_windows(scaler.transform(L[i]), FCFG["frames"], FCFG["window_hop"])[0] for i in idx]


@torch.no_grad()
def clip_scores(model, per_clip, device, batch=512):
    model.eval()
    flat = np.concatenate(per_clip)
    errs = []
    for i in range(0, len(flat), batch):
        x = torch.from_numpy(flat[i:i + batch][:, None]).to(device)
        errs.append(((model(x) - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
    errs = np.concatenate(errs)
    out, k = [], 0
    for w in per_clip:
        out.append(errs[k:k + len(w)].mean())
        k += len(w)
    return np.asarray(out)


def train_ae(L, tr, va, seed, device, log):
    set_seed(seed)
    scaler = BandStandardizer().fit([L[i] for i in tr], FCFG["std_eps"])
    Xtr = np.concatenate(windows(L, tr, scaler))[:, None]
    Wva = windows(L, va, scaler)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr)), batch_size=ACFG["batch_size"], shuffle=True, generator=g)
    model = ConvAutoencoder(FCFG["n_mels"], FCFG["frames"], ACFG["bottleneck"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ACFG["lr"])
    best, best_ep, state, wait, hist = np.inf, 0, None, 0, []
    for ep in range(1, ACFG["max_epochs"] + 1):
        model.train()
        tot = 0.0
        for (x,) in loader:
            x = x.to(device)
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        vl = float(clip_scores(model, Wva, device).mean())
        hist.append((ep, tot / len(Xtr), vl))
        if vl < best - 1e-6:
            best, best_ep, state, wait = vl, ep, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if wait >= ACFG["patience"]:
            break
    model.load_state_dict(state)
    log(f"    tohum {seed}: {len(tr)} klip / {len(Xtr)} pencere, {count_parameters(model):,} parametre, "
        f"en iyi epoch {best_ep}/{len(hist)}, doğrulama {best:.4f}")
    return model, scaler, dict(best_epoch=best_ep, epochs=len(hist), val_loss=best, history=hist)


# ---------------------------------------------------------------- katlar
def cond_order(tab):
    return [f"{r}r_{l}" for r, l in sorted(set(zip(tab.rpm, tab.load)))]


def split_fold(tab, h):
    n = tab.label == "normal"
    other = tab.cond != h
    return (np.where(n & other & (tab.part <= 3))[0], np.where(n & other & (tab.part == 4))[0],
            np.where(n & other & (tab.part == 5))[0], np.where(tab.cond == h)[0])


def run_folds(tab, Ls, out, seeds, device, log):
    sc_dir = out / "scores"
    sc_dir.mkdir(parents=True, exist_ok=True)
    B = {k: baseline_vectors(np.asarray(L)) for k, L in Ls.items()}
    for h in cond_order(tab):
        tr, va, ca, te = split_fold(tab, h)
        ev = np.r_[ca, te]
        role = np.r_[np.full(len(ca), "cal"), np.full(len(te), "test")]
        log(f"Kat {h}: eğitim {len(tr)}, doğrulama {len(va)}, kalibrasyon {len(ca)}, test {len(te)} klip")
        f = sc_dir / f"iforest_{h}.parquet"
        if not f.exists():
            pipe = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, random_state=42, n_jobs=-1))
            pipe.fit(B["temiz"][tr])
            rows = [pd.DataFrame(dict(idx=ev, role=role, noise=k, score=-pipe.score_samples(B[k][ev]))) for k in NOISES]
            pd.concat(rows).assign(method="iforest", seed=42, fold=h).to_parquet(f)
        for s in seeds:
            f = sc_dir / f"cnn_ae_s{s}_{h}.parquet"
            if f.exists():
                continue
            t0 = time.time()
            model, scaler, info = train_ae(Ls["temiz"], tr, va, s, device, log)
            rows = [pd.DataFrame(dict(idx=ev, role=role, noise=k,
                                      score=clip_scores(model, windows(Ls[k], ev, scaler), device))) for k in NOISES]
            pd.concat(rows).assign(method="cnn_ae", seed=s, fold=h).to_parquet(f)
            pd.DataFrame([dict(fold=h, seed=s, time_s=round(time.time() - t0, 1),
                               **{k: v for k, v in info.items() if k != "history"})]).to_csv(out / f"train_log_{h}_s{s}.csv", index=False)
    S = pd.concat([pd.read_parquet(p) for p in sorted(sc_dir.glob("*.parquet"))], ignore_index=True)
    return S.merge(tab[["label", "cond", "day"]], left_on="idx", right_index=True)


# ---------------------------------------------------------------- değerlendirme
def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def pauc(y, s):
    return roc_auc_score(y, s, max_fpr=0.1) if len(np.unique(y)) == 2 else np.nan


def evaluate(S):
    rows = []
    for (m, seed, fold), g in S.groupby(["method", "seed", "fold"]):
        thr_fixed = np.percentile(g[(g.role == "cal") & (g.noise == "temiz")].score, PERCENTILE)
        for nz in NOISES:
            t = g[(g.role == "test") & (g.noise == nz)]
            thr_re = np.percentile(g[(g.role == "cal") & (g.noise == nz)].score, PERCENTILE)
            nrm = t[t.label == "normal"].score.values
            for thr_name, thr in (("sabit", thr_fixed), ("yeniden", thr_re)):
                if nz == "temiz" and thr_name == "yeniden":
                    continue
                for fl in FAULTS + ["tümü"]:
                    fs = t[t.label != "normal"].score.values if fl == "tümü" else t[t.label == fl].score.values
                    y = np.r_[np.zeros(len(nrm)), np.ones(len(fs))]
                    sc = np.r_[nrm, fs]
                    tm = threshold_metrics(y, sc, thr)
                    rows.append(dict(method=m, seed=seed, fold=fold, noise=nz, threshold=thr_name, fault=fl,
                                     auc=auc(y, sc), pauc=pauc(y, sc), recall=tm["recall"], far=tm["far"],
                                     precision=tm["precision"], f1=tm["f1"], n_normal=len(nrm), n_fault=len(fs)))
    return pd.DataFrame(rows)


def paired_bootstrap(S, n_boot=300, seed=0):
    """AE (tohum ortalaması) − IF farkı, 9 kat × 4 arıza ortalama AUC; klipler kat ve sınıf içinde yeniden örneklenir."""
    T = S[(S.role == "test") & (S.noise == "temiz")]
    folds = sorted(T.fold.unique())
    seeds = sorted(T[T.method == "cnn_ae"].seed.unique())
    # her kat için skor matrisleri: satır = klip, sütun = [iforest, ae_s1, ae_s2, ...]
    data = {}
    for h in folds:
        t = T[T.fold == h]
        piv = t.pivot_table(index="idx", columns=["method", "seed"], values="score")
        cols = [("iforest", 42)] + [("cnn_ae", s) for s in seeds]
        lab = t.drop_duplicates("idx").set_index("idx").label.reindex(piv.index)
        data[h] = (piv[cols].values, lab.values)
    rng = np.random.default_rng(seed)

    def stat(sample):
        d_if, d_ae = [], []
        for h in folds:
            M, lab = data[h]
            ni = np.where(lab == "normal")[0]
            ni = sample(ni)
            for fl in FAULTS:
                fi = sample(np.where(lab == fl)[0])
                y = np.r_[np.zeros(len(ni)), np.ones(len(fi))]
                rows = np.r_[ni, fi]
                d_if.append(roc_auc_score(y, M[rows, 0]))
                d_ae.append(np.mean([roc_auc_score(y, M[rows, j]) for j in range(1, M.shape[1])]))
        return np.mean(d_if), np.mean(d_ae)

    point = stat(lambda a: a)
    boots = np.array([stat(lambda a: rng.choice(a, len(a))) for _ in range(n_boot)])
    diff = boots[:, 1] - boots[:, 0]
    return dict(auc_if=point[0], auc_ae=point[1], diff=point[1] - point[0],
                if_ci=np.percentile(boots[:, 0], [2.5, 97.5]).tolist(), ae_ci=np.percentile(boots[:, 1], [2.5, 97.5]).tolist(),
                diff_ci=np.percentile(diff, [2.5, 97.5]).tolist(), n_boot=n_boot)


def same_day(S):
    T = S[(S.role == "test") & (S.noise == "temiz")]
    rows = []
    for (m, seed, fold), t in T.groupby(["method", "seed", "fold"]):
        nrm = t[(t.label == "normal") & (t.day == SAME_DAY)].score.values
        fs = t[t.label == "outer"].score.values
        if len(nrm) == 0:
            continue
        y = np.r_[np.zeros(len(nrm)), np.ones(len(fs))]
        rows.append(dict(method=m, seed=seed, fold=fold, n_normal=len(nrm), auc=auc(y, np.r_[nrm, fs])))
    return pd.DataFrame(rows)


def summarize(M):
    """Önce kat ortalaması (tohum başına), sonra tohumlar arası ortalama ± sd ve katların en düşüğü."""
    per_seed = M.groupby(["method", "seed", "noise", "threshold", "fault"])[["auc", "pauc", "recall", "far", "f1"]].mean()
    agg = per_seed.groupby(["method", "noise", "threshold", "fault"]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    mins = M.groupby(["method", "seed", "noise", "threshold", "fault"]).auc.min().groupby(
        ["method", "noise", "threshold", "fault"]).mean().rename("auc_min_kat")
    return agg.join(mins).reset_index()


# ---------------------------------------------------------------- son (demo) modeller
def final_models(tab, Ls, ckpt, device, log):
    n = tab.label == "normal"
    tr, va, ca = (np.where(n & (tab.part <= 3))[0], np.where(n & (tab.part == 4))[0], np.where(n & (tab.part == 5))[0])
    B = baseline_vectors(np.asarray(Ls["temiz"]))
    base_meta = dict(machine_type="rulman", machine_id="tum_kosullar", features=FCFG,
                     protocol="9 koşulun normali: parça 1-3 eğitim, 4 doğrulama, 5 kalibrasyon")
    d = ckpt / "iforest" / "rulman" / "tum_kosullar"
    d.mkdir(parents=True, exist_ok=True)
    pipe = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)).fit(B[tr])
    joblib.dump(pipe, d / "model.joblib")
    save_json(dict(method="iforest", seed=42, **base_meta), d / "meta.json")
    thr = float(np.percentile(-pipe.score_samples(B[ca]), PERCENTILE))
    save_json(dict(threshold=thr, percentile=PERCENTILE, n_cal=len(ca)), d / "threshold.json")
    log(f"Son IF modeli → {d} (eşik {thr:.4f})")

    model, scaler, info = train_ae(Ls["temiz"], tr, va, 42, device, log)
    d = ckpt / "cnn_ae" / "seed42" / "rulman" / "tum_kosullar"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "best.pt")
    scaler.save(d / "scaler.npz")
    save_json(dict(method="cnn_ae", seed=42, bottleneck=ACFG["bottleneck"], aggregation="mean",
                   best_epoch=info["best_epoch"], best_val_loss=info["val_loss"], **base_meta), d / "meta.json")
    thr = float(np.percentile(clip_scores(model, windows(Ls["temiz"], ca, scaler), device), PERCENTILE))
    save_json(dict(threshold=thr, percentile=PERCENTILE, n_cal=len(ca)), d / "threshold.json")
    log(f"Son CNN AE modeli → {d} (eşik {thr:.5f})")


# ---------------------------------------------------------------- rapor
def fmt(x, nd=3):
    return "—" if pd.isna(x) else f"{x:.{nd}f}".replace(".", ",")


def write_summary(out, tab, Msum, boot, sd, Mfold, seeds):
    nk = tab.cond.nunique()
    L = ["# Guilin akustik rulman veri seti — sonuçlar", "",
         f"Koşul-dışı bırakma ({nk} kat). CNN AE: {len(seeds)} tohum ({', '.join(map(str, seeds))}) ortalaması ± tohumlar arası sd; "
         "IF: tohum 42. Eşik: eğitim koşullarının normal kalibrasyon kliplerinin 95. persentili.", "",
         f"Veri: {len(tab)} klip (10 sn) — " + ", ".join(f"{k} {v}" for k, v in tab.label.value_counts().items()), ""]
    c = Msum[(Msum.noise == "temiz")]
    L += [f"## Temiz test — arıza türü başına ({nk} kat ortalaması)", "",
          "| Yöntem | Arıza | AUC | pAUC | En düşük kat AUC | Yakalama | Yanlış alarm | F1 |", "|---|---|---|---|---|---|---|---|"]
    for m in ("iforest", "cnn_ae"):
        for fl in FAULTS + ["tümü"]:
            r = c[(c.method == m) & (c.fault == fl)].iloc[0]
            pm = lambda k: fmt(r[f"{k}_mean"]) + ("" if m == "iforest" else f" ± {fmt(r[f'{k}_std'])}")
            L.append(f"| {METHOD_TR[m]} | {FAULT_TR[fl]} | {pm('auc')} | {pm('pauc')} | {fmt(r['auc_min_kat'])} | "
                     f"{pm('recall')} | {pm('far')} | {pm('f1')} |")
    L += ["", "## Ortalama AUC farkı (CNN AE − IF), eşleştirilmiş bootstrap", "",
          f"- IF: {fmt(boot['auc_if'])} [{fmt(boot['if_ci'][0])}; {fmt(boot['if_ci'][1])}]",
          f"- CNN AE: {fmt(boot['auc_ae'])} [{fmt(boot['ae_ci'][0])}; {fmt(boot['ae_ci'][1])}]",
          f"- Fark: **{fmt(boot['diff'])} [{fmt(boot['diff_ci'][0])}; {fmt(boot['diff_ci'][1])}]** "
          f"(%95 GA, {boot['n_boot']} tekrar; {nk} kat × 4 arıza türü ortalaması)", ""]
    k = Mfold[(Mfold.noise == "temiz") & (Mfold.fault == "tümü")].groupby(["method", "fold"])[["auc", "far", "recall"]].mean().unstack(0)
    L += ["## Kat (dışarıda bırakılan koşul) başına — tüm arızalar", "",
          "| Koşul | IF AUC | CNN AE AUC | IF yanlış alarm | CNN AE yanlış alarm | IF yakalama | CNN AE yakalama |", "|---|---|---|---|---|---|---|"]
    for h in cond_order(tab):
        r = k.loc[h]
        L.append(f"| {h.replace('r_', ' rpm, %')} yük | {fmt(r[('auc', 'iforest')])} | {fmt(r[('auc', 'cnn_ae')])} | "
                 f"{fmt(r[('far', 'iforest')])} | {fmt(r[('far', 'cnn_ae')])} | {fmt(r[('recall', 'iforest')])} | {fmt(r[('recall', 'cnn_ae')])} |")
    if len(sd):
        s = sd.groupby("method").agg(auc=("auc", "mean"), kat=("fold", "nunique"), n=("n_normal", "sum"))
        L += ["", f"## Aynı gün alt kümesi ({SAME_DAY}: normal ve dış bilezik aynı gün kaydedildi)", "",
              "| Yöntem | AUC (kat ortalaması) | Kat sayısı |", "|---|---|---|"]
        for m, r in s.iterrows():
            L.append(f"| {METHOD_TR[m]} | {fmt(r.auc)} | {int(r.kat)} |")
    g = Msum[Msum.fault == "tümü"]
    L += ["", "## Gürültü dayanıklılığı (tüm arızalar)", "",
          "| Yöntem | Koşul | Eşik | AUC | Yakalama | Yanlış alarm |", "|---|---|---|---|---|---|"]
    for m in ("iforest", "cnn_ae"):
        for nz in NOISES:
            for th in ("sabit", "yeniden"):
                r = g[(g.method == m) & (g.noise == nz) & (g.threshold == th)]
                if len(r):
                    r = r.iloc[0]
                    L.append(f"| {METHOD_TR[m]} | {'temiz' if nz == 'temiz' else '+' + nz + ' dB'} | "
                             f"{'özgün kalibrasyon' if th == 'sabit' else 'gürültülü kalibrasyon'} | {fmt(r.auc_mean)} | "
                             f"{fmt(r.recall_mean)} | {fmt(r.far_mean)} |")
    L += ["", "Not: Normal ve arızalı kayıtlar aynı oturumda alınmamıştır; aynı gün alt kümesi bu etkiyi kısmen sınar.", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--skip-final", action="store_true")
    a = ap.parse_args()
    if a.max_epochs:
        ACFG["max_epochs"] = a.max_epochs
    wav, out, ckpt = Path(a.wav), Path(a.out), Path(a.ckpt)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.time()
    tab = load_table(wav)
    log(f"{len(tab)} klip, {tab.cond.nunique()} koşul, cihaz {device}")
    Ls = {k: logmel_cache(tab, wav, out, k) for k in NOISES}
    log(f"Log mel önbelleği hazır: {Ls['temiz'].shape} · {time.time() - t0:.0f} s")
    S = run_folds(tab, Ls, out, a.seeds, device, log)
    Mfold = evaluate(S)
    Mfold.to_csv(out / "metrics_fold.csv", index=False)
    Msum = summarize(Mfold)
    Msum.to_csv(out / "metrics_summary.csv", index=False)
    boot = paired_bootstrap(S, a.n_boot)
    save_json(boot, out / "bootstrap.json")
    sd = same_day(S)
    sd.to_csv(out / "same_day.csv", index=False)
    write_summary(out, tab, Msum, boot, sd, Mfold, a.seeds)
    tl = sorted(out.glob("train_log_*.csv"))
    if tl:
        pd.concat([pd.read_csv(p) for p in tl]).to_csv(out / "train_log.csv", index=False)
    if not a.skip_final:
        final_models(tab, Ls, ckpt, device, log)
    save_json(dict(fcfg=FCFG, acfg=ACFG, seeds=a.seeds, percentile=PERCENTILE, noise_seed=NOISE_SEED,
                   total_time_s=round(time.time() - t0)), out / "run_config.json")
    log(f"Bitti · {time.time() - t0:.0f} s → {out / 'summary.md'}")


if __name__ == "__main__":
    main()
