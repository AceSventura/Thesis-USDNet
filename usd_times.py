import os
import glob
import time
import argparse

import numpy as np
import pandas as pd
import torch
import tifffile
from tqdm import tqdm

from lapsrn import Mpattern_opt
from Spectral_demosaicing import input_matrix_wpn

# ==============================================================================
# 0. CONFIGURAZIONE
# ==============================================================================
MSFA_SIZE = 3

PATTERN = np.array([
    [0, 5, 2],
    [3, 8, 7],
    [6, 1, 4],
])


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/mnt/Volume_Interno/UBUNTU/dataset_flavia/valid")
    p.add_argument("--usd_ckpt",
                   default="checkpoint/Flavia_V2_LSA_3_EItrain_Transrandom_alpha1_st1_261003_103359/De_happy_model_epoch_2190.pth")
    p.add_argument("--tile_size", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--warmup_runs", type=int, default=3, help="Esecuzioni di hot start PER OGNI immagine")
    p.add_argument("--n_runs", type=int, default=20, help="Esecuzioni misurate PER OGNI immagine")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--out_dir", default="results/benchmark")
    return p.parse_args()


# ==============================================================================
# 1. CARICAMENTO DATI (una immagine alla volta, FUORI dal tempo misurato)
# ==============================================================================
def load_sample(sample_dir: str, msfa_pattern: np.ndarray, device):
    sparse_dir = os.path.join(sample_dir, "sparse_matrices")
    sparse_list = []
    for band_idx in msfa_pattern.flatten():
        f = os.path.join(sparse_dir, f"sparse_matrix_{band_idx}.tiff")
        sparse_list.append(tifffile.imread(f).astype(np.float32) / 255.0)
    sparse = np.stack(sparse_list, axis=0)  # C,H,W

    mosaic_files = glob.glob(os.path.join(sample_dir, "*Mosaic*.tif*"))
    raw = tifffile.imread(mosaic_files[0]).astype(np.float32) / 255.0

    H, W = raw.shape
    m0 = (H // MSFA_SIZE) * MSFA_SIZE
    m1 = (W // MSFA_SIZE) * MSFA_SIZE
    sparse = np.ascontiguousarray(sparse[:, :m0, :m1])
    raw = np.ascontiguousarray(raw[:m0, :m1])

    return {
        "raw": torch.from_numpy(raw)[None, None].to(device),
        "sparse": torch.from_numpy(sparse)[None].to(device),
    }


# ==============================================================================
# 2. MODELLO USD
# ==============================================================================
def load_usd(path, device):
    model = Mpattern_opt(MSFA_SIZE, "LSA").to(device)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # versioni di torch senza weights_only
        ckpt = torch.load(path, map_location=device)
    m = ckpt["model"]
    model.load_state_dict(m.state_dict() if hasattr(m, "state_dict") else m)
    return model.eval()


# Cache delle matrici posizionali per forma di tile: viene riempita durante il
# warm-up e riusata, quindi il calcolo di input_matrix_wpn NON rientra nel
# tempo misurato (in deployment si precalcola una sola volta).
_POS_CACHE = {}


def get_pos(th, tw, device):
    key = (th, tw, str(device))
    if key not in _POS_CACHE:
        _POS_CACHE[key] = input_matrix_wpn(th, tw, MSFA_SIZE).to(device).float()
    return _POS_CACHE[key]


@torch.inference_mode()
def forward_tiled_usd(model, raw, sparse, tile_size, overlap):
    N, C, H, W = sparse.shape
    tile_size = tile_size - (tile_size % MSFA_SIZE)
    overlap = overlap - (overlap % MSFA_SIZE)
    stride = tile_size - overlap
    half = overlap // 2
    out = torch.zeros((N, C, H, W), dtype=torch.float32, device=sparse.device)

    for h in range(0, H, stride):
        for w in range(0, W, stride):
            h_end, w_end = min(h + tile_size, H), min(w + tile_size, W)
            th, tw = h_end - h, w_end - w
            pos = get_pos(th, tw, sparse.device)
            full_tile = model([sparse[:, :, h:h_end, w:w_end],
                               raw[:, :, h:h_end, w:w_end]], pos)

            ct = 0 if h == 0 else half
            cb = th if h_end == H else th - half
            cl = 0 if w == 0 else half
            cr = tw if w_end == W else tw - half
            out[:, :, h + ct:h + cb, w + cl:w + cr] = full_tile[:, :, ct:cb, cl:cr]
    return out


# ==============================================================================
# 3. MISURE SU SINGOLA IMMAGINE
# ==============================================================================
def sync(device):
    torch.cuda.synchronize(device)


def time_runs(run_fn, sample, n_runs, device):
    ts = []
    for _ in range(n_runs):
        sync(device)
        t0 = time.perf_counter()
        out = run_fn(sample)
        sync(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
        del out
    return np.array(ts)


def peak_extra_mb(run_fn, sample, device):
    """Picco di memoria GPU aggiuntiva (pesi e input esclusi, output incluso)."""
    torch.cuda.empty_cache()
    sync(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = run_fn(sample)
    sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    del out
    return (peak - base) / 1024 ** 2


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA non disponibile: il benchmark richiede una GPU.")
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    sample_dirs = sorted([d.path for d in os.scandir(args.data_dir) if d.is_dir()])
    if args.max_images:
        sample_dirs = sample_dirs[:args.max_images]

    model = load_usd(args.usd_ckpt, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"USD parametri: {params:,}")

    def run_fn(s):
        return forward_tiled_usd(model, s["raw"], s["sparse"],
                                 args.tile_size, args.overlap).clamp(0, 1)

    rows = []
    for sample_dir in tqdm(sample_dirs, desc="Benchmark USD per immagine"):
        name = os.path.basename(sample_dir)
        sample = load_sample(sample_dir, PATTERN, device)
        H, W = sample["raw"].shape[-2:]

        for _ in range(args.warmup_runs):
            o = run_fn(sample)
            del o
        sync(device)

        mem = peak_extra_mb(run_fn, sample, device)
        ts = time_runs(run_fn, sample, args.n_runs, device)

        rows.append({
            "Filename": name,
            "Resolution": f"{H}x{W}",
            "Mpx": H * W / 1e6,
            "Method": "USD",
            "Mean_ms": ts.mean(),
            "Std_ms": ts.std(ddof=1) if len(ts) > 1 else 0.0,
            "Median_ms": float(np.median(ts)),
            "Min_ms": ts.min(),
            "Max_ms": ts.max(),
            "Peak_extra_MB": mem,
        })
        del sample
        torch.cuda.empty_cache()

    df_img = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "Method": "USD",
        "Params": params,
        "Time_ms_mean": df_img.Mean_ms.mean(),
        "Time_ms_std_across_images": df_img.Mean_ms.std(),
        "RunToRun_std_ms": df_img.Std_ms.mean(),
        "Median_ms": df_img.Median_ms.mean(),
        "Min_ms": df_img.Min_ms.mean(),
        "Peak_extra_MB_mean": df_img.Peak_extra_MB.mean(),
        "Peak_extra_MB_max": df_img.Peak_extra_MB.max(),
        "N_images": len(df_img),
        "Throughput_Mpx_per_s": df_img.Mpx.mean() / (df_img.Mean_ms.mean() / 1000.0),
    }]).round(4)

    os.makedirs(args.out_dir, exist_ok=True)
    df_img.round(4).to_csv(os.path.join(args.out_dir, "timing_usd_per_image_gpu.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "timing_usd_summary_gpu.csv"), index=False)

    print("\n" + summary.to_string(index=False))
    print(f"\nSalvato in: {args.out_dir}/timing_usd_per_image_gpu.csv e timing_usd_summary_gpu.csv")


if __name__ == "__main__":
    main()