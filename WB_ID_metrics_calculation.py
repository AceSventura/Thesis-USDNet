import os
import glob
import numpy as np
import pandas as pd
import tifffile
from joblib import Parallel, delayed
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim_sk

# Si assume l'esistenza del modulo esterno per ERGAS come nel file originale
from Spectral_demosaicing import ergas_matlab

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

def compute_all_metrics(true_img: np.ndarray, pred_img: np.ndarray) -> dict:
    """
    Riceve i crop già scalati e arrotondati in [0, 255] come uint8.
    """
    num_bands = true_img.shape[2]
    win_size = min(7, true_img.shape[0], true_img.shape[1])
    if win_size % 2 == 0:
        win_size -= 1

    psnr_vals = [
        peak_signal_noise_ratio(true_img[:, :, b], pred_img[:, :, b], data_range=255)
        for b in range(num_bands)
    ]
    
    ssim_vals = Parallel(n_jobs=-1)(
        delayed(ssim_sk)(pred_img[:, :, b], true_img[:, :, b], data_range=255, win_size=win_size)
        for b in range(num_bands)
    )

    return {
        "PSNR": float(np.mean(psnr_vals)),
        "SSIM": float(np.mean(ssim_vals)),
        "SAM": sam_metric(true_img.astype(np.float64), pred_img.astype(np.float64)),
        "ERGAS": ergas_matlab(true_img, pred_img, 1/3)
    }

# ==============================================================================
# 2. GENERAZIONE MASCHERA (Da File 2)
# ==============================================================================
def generate_mask(img_shape, band_idx):
    H, W = img_shape
    mask = np.zeros(img_shape, dtype=bool)

    # Mappa banda scelta -> indice maschera
    band_to_mask = [0, 5, 2, 3, 8, 7, 6, 1, 4]
    mask_idx = band_to_mask[band_idx]

    # shift colonne: 0,1,2 ciclico
    col_shift = mask_idx % 3
    valid_cols = np.arange(col_shift, W, 3)

    # determinazione righe secondo pattern di 3 righe (0,1,2)
    row_pattern = mask_idx // 3
    valid_rows = np.arange(row_pattern, H, 3)

    # impostazione pixel validi
    mask[valid_rows[:, None], valid_cols] = True
    return mask

# ==============================================================================
# 3. FUNZIONI DI BASELINE (WB e ID - Da File 2)
# ==============================================================================
def conv_WB_baseline(sparse_cube: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    WB_matrices = np.zeros_like(sparse_cube, dtype=np.float64)
    for band in range(sparse_cube.shape[2]):
        WB_matrices[:, :, band] = convolve2d(sparse_cube[:, :, band], kernel, mode='same')
    return WB_matrices

def intensity_interpolation(full_mosaic_image: np.ndarray, m_l: np.ndarray, kernel: np.ndarray, num_bands: int) -> np.ndarray:
    H, W = full_mosaic_image.shape
    ID_matrices = np.zeros((H, W, num_bands), dtype=np.float64)
    
    M = (1/9) * np.ones((3, 3))
    I_M = convolve(full_mosaic_image, M, mode='reflect')

    for k in range(num_bands):
        Delta_k = (full_mosaic_image - I_M) * m_l[:, :, k]
        Delta_k_stim = convolve(Delta_k, kernel, mode='reflect')
        ID_matrices[:, :, k] = I_M + Delta_k_stim
        
    return ID_matrices

# ==============================================================================
# 4. CARICAMENTO DATI (Allineato a File 2: Lettura float [0,1])
# ==============================================================================
def load_sample_from_folder(sample_dir: str, num_bands: int = 9) -> tuple:
    orig_dir = os.path.join(sample_dir, 'original_bands')
    sparse_dir = os.path.join(sample_dir, 'sparse_matrices')

    target_list, sparse_list = [], []
    for band_idx in range(num_bands):
        orig_file = os.path.join(orig_dir, f'original_band_{band_idx}.tiff')
        sparse_file = os.path.join(sparse_dir, f'sparse_matrix_{band_idx}.tiff')
        
        # Lettura immagine normalizzata [0,1]
        target_img = tifffile.imread(orig_file).astype(float) / 255.0
        sparse_img = tifffile.imread(sparse_file).astype(float) / 255.0
        
        target_list.append(target_img)
        sparse_list.append(sparse_img)

    target = np.stack(target_list, axis=-1)
    sparse_cube = np.stack(sparse_list, axis=-1)

    mosaic_files = glob.glob(os.path.join(sample_dir, '*Mosaic*.tif*'))
    raw = tifffile.imread(mosaic_files[0]).astype(float) / 255.0

    return target, sparse_cube, raw

# ==============================================================================
# 5. PIPELINE PRINCIPALE
# ==============================================================================
def main():
    test_dir = "/mnt/Volume_Interno/UBUNTU/dataset_flavia/test"
    output_csv = "./results/benchmark/results_wb_id_fabio.csv"
    num_bands = 9
    
    # Kernel di convoluzione (5x5) esattamente come in file 2
    base_vector = np.array([1, 2, 3, 2, 1])
    kernel = (1/3) * np.outer(base_vector, base_vector) * (1/3)
    
    results = []
    sample_dirs = sorted([d.path for d in os.scandir(test_dir) if d.is_dir()])
    
    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)
        
        # Caricamento dati in formato [0, 1]
        gt_np, sparse_np, raw_np = load_sample_from_folder(sample_dir, num_bands)
        H, W = raw_np.shape
        
        # Generazione analitica delle maschere booleane
        masks_np = np.zeros((H, W, num_bands), dtype=bool)
        for k in range(num_bands):
            masks_np[:, :, k] = generate_mask((H, W), k)
        
        # Inferenza Baseline (Output range [0, 1])
        wb_pred_norm = conv_WB_baseline(sparse_np, kernel)
        id_pred_norm = intensity_interpolation(raw_np, masks_np, kernel, num_bands)
        
        # Denormalizzazione in [0, 255] e cast a uint8
        gt_denorm = np.clip(gt_np * 255, 0, 255).round().astype(np.uint8)
        wb_pred = np.clip(wb_pred_norm * 255, 0, 255).round().astype(np.uint8)
        id_pred = np.clip(id_pred_norm * 255, 0, 255).round().astype(np.uint8)
        
        # Logica di ritaglio analitica
        m0 = int(np.fix(H / 3) * 3)
        m1 = int(np.fix(W / 3) * 3)
        
        gt_crop = gt_denorm[2 : (m0 - 3), 2 : (m1 - 3)]
        wb_crop = wb_pred[2 : (m0 - 3), 2 : (m1 - 3)]
        id_crop = id_pred[2 : (m0 - 3), 2 : (m1 - 3)]
        
        # Calcolo Metriche
        wb_metrics = compute_all_metrics(gt_crop, wb_crop)
        id_metrics = compute_all_metrics(gt_crop, id_crop)
        
        # Compilazione record
        results.append({
            "Filename": sample_name,
            "Resolution": f"{gt_crop.shape[0]}x{gt_crop.shape[1]}",
            "WB_PSNR": round(wb_metrics["PSNR"], 4),
            "WB_SSIM": round(wb_metrics["SSIM"], 4),
            "WB_SAM": round(wb_metrics["SAM"], 4),
            "WB_ERGAS": round(wb_metrics["ERGAS"], 4),
            "ID_PSNR": round(id_metrics["PSNR"], 4),
            "ID_SSIM": round(id_metrics["SSIM"], 4),
            "ID_SAM": round(id_metrics["SAM"], 4),
            "ID_ERGAS": round(id_metrics["ERGAS"], 4)
        })
        print(f"Elaborato: {sample_name}")

    # Aggregazione e Salvataggio
    if results:
        df = pd.DataFrame(results)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        mean_row = {col: round(df[col].mean(), 4) for col in numeric_cols}
        mean_row["Filename"] = "MEDIA"
        mean_row["Resolution"] = "-"
        
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Salvataggio completato in: {output_csv}")

if __name__ == "__main__":
    main()