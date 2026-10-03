import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import tifffile
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim_sk

# Import relativi al tuo modello
from lapsrn import Mpattern_opt
from Spectral_demosaicing import input_matrix_wpn, ergas_matlab

# ==============================================================================
# 1. FUNZIONI DI METRICA
# ==============================================================================
def sam_metric(gt_cube: np.ndarray, pred: np.ndarray) -> float:
    """Calcola il SAM medio in gradi su tensori (H, W, C)."""
    dot_product = np.sum(gt_cube * pred, axis=2)
    gt_norm = np.linalg.norm(gt_cube, axis=2)
    pred_norm = np.linalg.norm(pred, axis=2)
    denom = gt_norm * pred_norm
    
    valid = denom > 0
    cosine = np.zeros_like(denom)
    cosine[valid] = dot_product[valid] / denom[valid]
    
    angles = np.degrees(np.arccos(np.clip(cosine[valid], -1.0, 1.0)))
    return float(np.mean(angles)) if angles.size else 0.0

def compute_metrics_cropped(gt_crop: np.ndarray, pred_crop: np.ndarray) -> dict:
    """Calcola le metriche riportando i dati alla scala [0,255] in interi per equità con WB/ID."""
    true_img = np.clip(np.round(gt_crop * 255.0), 0, 255).astype(np.uint8)
    pred_img = np.clip(np.round(pred_crop * 255.0), 0, 255).astype(np.uint8)
    
    C = true_img.shape[2]
    win_size = min(7, true_img.shape[0], true_img.shape[1])
    if win_size % 2 == 0:
        win_size -= 1

    psnr_vals = [
        peak_signal_noise_ratio(true_img[:, :, b], pred_img[:, :, b], data_range=255)
        for b in range(C)
    ]
    ssim_vals = Parallel(n_jobs=-1)(
        delayed(ssim_sk)(pred_img[:, :, b], true_img[:, :, b], data_range=255, win_size=win_size)
        for b in range(C)
    )

    return {
        "PSNR":  float(np.mean(psnr_vals)),
        "SSIM":  float(np.mean(ssim_vals)),
        "SAM":   sam_metric(true_img.astype(np.float64), pred_img.astype(np.float64)),
        "ERGAS": ergas_matlab(true_img, pred_img, 1/3),
    }

# ==============================================================================
# 2. LETTURA DATI PER IL MODELLO (Basata sul Pattern Spaziale Reale)
# ==============================================================================
def load_sample_from_folder(sample_dir: str, msfa_pattern: np.ndarray) -> tuple:
    orig_dir = os.path.join(sample_dir, 'original_bands')
    sparse_dir = os.path.join(sample_dir, 'sparse_matrices')
    
    # Appiattimento del pattern: l'ordine di impilamento sarà [0, 5, 2, 3, 8, 7, 6, 1, 4]
    flat_pattern = msfa_pattern.flatten()

    target_list, sparse_list = [], []
    for band_idx in flat_pattern:
        orig_file = os.path.join(orig_dir, f'original_band_{band_idx}.tiff')
        sparse_file = os.path.join(sparse_dir, f'sparse_matrix_{band_idx}.tiff')
        
        target_img = tifffile.imread(orig_file).astype(np.float32) / 255.0
        sparse_img = tifffile.imread(sparse_file).astype(np.float32) / 255.0
        
        target_list.append(target_img)
        sparse_list.append(sparse_img)

    target = np.stack(target_list, axis=-1)
    sparse_cube = np.stack(sparse_list, axis=-1)

    mosaic_files = glob.glob(os.path.join(sample_dir, '*Mosaic*.tif*'))
    raw = tifffile.imread(mosaic_files[0]).astype(np.float32) / 255.0

    return target, sparse_cube, raw

# ==============================================================================
# 3. FUNZIONE DI FORWARD CON TILING
# ==============================================================================
def forward_tiled_usd(model, sparse_tensor: torch.Tensor, raw_tensor: torch.Tensor, msfa_size: int, tile_size: int, overlap: int) -> np.ndarray:
    device = sparse_tensor.device
    N, C, H, W = sparse_tensor.shape
    tile_size = tile_size - (tile_size % msfa_size)
    overlap = overlap - (overlap % msfa_size)
    stride = tile_size - overlap
    half_overlap = overlap // 2
    out_full = torch.zeros((N, C, H, W), dtype=torch.float32, device=device)
    pos_cache = {}

    for h in range(0, H, stride):
        for w in range(0, W, stride):
            h_end, w_end = min(h + tile_size, H), min(w + tile_size, W)
            th, tw = h_end - h, w_end - w
            sparse_tile = sparse_tensor[:, :, h:h_end, w:w_end]
            raw_tile = raw_tensor[:, :, h:h_end, w:w_end]

            key = (th, tw)
            if key not in pos_cache:
                pos_cache[key] = input_matrix_wpn(th, tw, msfa_size).to(device).float()

            with torch.no_grad():
                full_tile = model([sparse_tile, raw_tile], pos_cache[key])

            crop_top = 0 if h == 0 else half_overlap
            crop_bottom = th if h_end == H else th - half_overlap
            crop_left = 0 if w == 0 else half_overlap
            crop_right = tw if w_end == W else tw - half_overlap

            out_full[:, :, h + crop_top:h + crop_bottom, w + crop_left:w + crop_right] = \
                full_tile[:, :, crop_top:crop_bottom, crop_left:crop_right].cpu()

    return out_full.squeeze(0).cpu().numpy().transpose(1, 2, 0)

# ==============================================================================
# 4. PIPELINE PRINCIPALE DI VALUTAZIONE
# ==============================================================================
def main():
    msfa_size = 3
    test_dir = "/mnt/Volume_Interno/UBUNTU/dataset_flavia/test"
    checkpoint_usd = "checkpoint/Flavia_V2_LSA_3_EItrain_Transrandom_alpha1_st1_261003_103359/De_happy_model_epoch_2190.pth"
    output_csv = "results/benchmark/USD_Evaluation_V2_LAST_2190_test.csv"
        
    # Pattern rigoroso allineato alla disposizione spaziale fisica per il Dataloader
    pattern = np.array([
        [0, 5, 2],
        [3, 8, 7],
        [6, 1, 4]
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Caricamento Modello...")

    model = Mpattern_opt(msfa_size, 'LSA').to(device)
    model.load_state_dict(torch.load(checkpoint_usd, map_location=device)["model"].state_dict())
    model.eval()

    sample_dirs = sorted([d.path for d in os.scandir(test_dir) if d.is_dir()])
    results = []

    for sample_dir in tqdm(sample_dirs, desc="Valutazione CNN"):
        sample_name = os.path.basename(sample_dir)
        
        # Caricamento e normalizzazione in [0, 1]
        img_np, sparse_np_full, raw_np_full = load_sample_from_folder(sample_dir, pattern)
        H_orig, W_orig, _ = img_np.shape

        # Mantenimento margini spaziali multipli di 3
        m0 = int(np.fix(H_orig / 3) * 3)
        m1 = int(np.fix(W_orig / 3) * 3)
        
        sparse_np_net = sparse_np_full[:m0, :m1, :]
        raw_np_net = raw_np_full[:m0, :m1]
        
        # Trasformazione in tensori PyTorch (H,W,C -> C,H,W)
        sparse_tensor = torch.from_numpy(np.ascontiguousarray(sparse_np_net.transpose(2, 0, 1))).unsqueeze(0).to(device)
        raw_tensor = torch.from_numpy(raw_np_net).unsqueeze(0).unsqueeze(0).to(device)

        # Inferenza
        usd_pred = forward_tiled_usd(model, sparse_tensor, raw_tensor, msfa_size=msfa_size, tile_size=1200, overlap=60)
        usd_pred = np.clip(usd_pred, 0.0, 1.0)

        # Ritaglio dinamico speculare:
        # Modificato a - 3 per allineare l'output a WB e ID.
        gt_crop = img_np[2 : m0 - 3, 2 : m1 - 3]
        usd_crop = usd_pred[2 : m0 - 3, 2 : m1 - 3]

        # Calcolo metriche
        metrics = compute_metrics_cropped(gt_crop, usd_crop)

        eval_H, eval_W = gt_crop.shape[:2]
        results.append({
            "Filename": sample_name,
            "Resolution_Eval": f"{eval_H}x{eval_W}",
            "USD_PSNR": round(metrics["PSNR"], 4),
            "USD_SSIM": round(metrics["SSIM"], 4),
            "USD_SAM": round(metrics["SAM"], 4),
            "USD_ERGAS": round(metrics["ERGAS"], 4)
        })

    # Aggregazione e Salvataggio
    if results:
        df = pd.DataFrame(results)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        mean_row = {col: round(df[col].mean(), 4) for col in numeric_cols}
        mean_row["Filename"] = "MEDIA GLOBALE"
        mean_row["Resolution_Eval"] = "-"

        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSalvataggio metriche completato in: {output_csv}")

if __name__ == "__main__":
    main()