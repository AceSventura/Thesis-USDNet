import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from GenericFolderDataset import GenericFolderDataset   # <-- adatta il nome del modulo
from Spectral_demosaicing import input_matrix_wpn as input_matrix_wpn_msfasize
from Spectral_demosaicing import pixel_shuffle_inv


def forward_tiled_usd(model, sparse_tensor, raw_tensor, msfa_size, tile_size=250, overlap=50):
    device = sparse_tensor.device
    N, C, H, W = sparse_tensor.shape

    tile_size = tile_size - (tile_size % msfa_size)
    overlap = overlap - (overlap % msfa_size)
    stride = tile_size - overlap
    half_overlap = overlap // 2

    out_full = torch.zeros((N, C, H, W), dtype=torch.float32, device='cpu')
    pos_cache = {}

    for h in range(0, H, stride):
        for w in range(0, W, stride):
            h_end = min(h + tile_size, H)
            w_end = min(w + tile_size, W)
            th, tw = h_end - h, w_end - w

            sparse_tile = sparse_tensor[:, :, h:h_end, w:w_end]
            raw_tile = raw_tensor[:, :, h:h_end, w:w_end]

            key = (th, tw)
            if key not in pos_cache:
                pos_cache[key] = input_matrix_wpn_msfasize(th, tw, msfa_size).to(device).float()

            with torch.no_grad():
                full_tile = model([sparse_tile, raw_tile], pos_cache[key])

            crop_top = 0 if h == 0 else half_overlap
            crop_bottom = th if h_end == H else th - half_overlap
            crop_left = 0 if w == 0 else half_overlap
            crop_right = tw if w_end == W else tw - half_overlap

            out_full[:, :, h + crop_top:h + crop_bottom, w + crop_left:w + crop_right] = \
                full_tile[:, :, crop_top:crop_bottom, crop_left:crop_right].cpu()

    return out_full.to(device)


def compute_sei(pred, msfa_size):
    """pred: array (C, H, W) in [0,1]. Ritorna SEI per banda (C,)."""
    sei_bands = np.zeros(msfa_size ** 2)
    for bn in range(msfa_size ** 2):
        band = pixel_shuffle_inv(np.expand_dims(np.expand_dims(pred[bn], 0), 0), msfa_size)
        band = np.mean(np.mean(band, -1), -1)
        sei_bands[bn] = float(np.asarray(band.var(axis=1)).reshape(-1)[0])
    return sei_bands


# ------------------------------------------------------------------ setup
parser = argparse.ArgumentParser(description="SEI per checkpoint")
parser.add_argument("--msfa_size", default=3, type=int)
parser.add_argument("--val_path", default="/mnt/Volume_Interno/UBUNTU/dataset_flavia/valid/", type=str)
parser.add_argument("--save_path", default="results/syn/", type=str)
parser.add_argument("--tile_size", default=600, type=int)
parser.add_argument("--overlap", default=30, type=int)
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(opt.save_path, exist_ok=True)

if opt.msfa_size == 3:
    msfa_pattern = np.array([[0, 5, 2],
                             [3, 8, 7],
                             [6, 1, 4]])
else:
    msfa_pattern = np.arange(opt.msfa_size ** 2).reshape(opt.msfa_size, opt.msfa_size)

# Dataset di validazione: immagini intere (patch_size=None), niente augmentation, normalizzazione /255
val_set = GenericFolderDataset(
    image_dir=opt.val_path,
    msfa_size=opt.msfa_size,
    msfa_pattern=msfa_pattern.flatten(),
    norm_flag=True,
    patch_size=None,
    augment=False,
    type='val',
)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=2, pin_memory=False)
print(f"Immagini di validazione: {len(val_set)}")

type_name_list = ['Flavia_V2_LSA_3_EItrain_Transrandom_alpha1_st1_261003_103359']
periodic_avg_dict = OrderedDict()

# ------------------------------------------------------------------ loop checkpoint
for type_name in type_name_list:
    for epoch_num in range(2000, 4000, 10):
        ckpt = f"checkpoint/{type_name}/De_happy_model_epoch_{epoch_num}.pth"
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} non trovato")
            continue

        print(ckpt)
        model = torch.load(ckpt, weights_only=False)["model"].to(device).eval()

        sei_sum = np.zeros(opt.msfa_size ** 2)
        n_samples = 0

        with torch.no_grad():
            for raw, sparse, _target in val_loader:
                raw = raw.float().to(device)        # (1, 1, H, W)
                sparse = sparse.float().to(device)  # (1, C, H, W)

                out = forward_tiled_usd(model, sparse, raw, opt.msfa_size,
                                        tile_size=opt.tile_size, overlap=opt.overlap)
                pred = out[0].cpu().numpy().astype(np.float32)   # (C, H, W) in [0,1]

                sei_sum += compute_sei(pred, opt.msfa_size)
                n_samples += 1

                del out, raw, sparse

        sei_bands = sei_sum / n_samples
        sei_mean = sei_bands.mean()
        print(f"--- Epoch {epoch_num} | SEI medio = {sei_mean:.6e}")
        for b, v in enumerate(sei_bands):
            print(f"   Banda {b:2d} -> SEI: {v:.6e}")

        periodic_avg_dict[epoch_num] = np.concatenate(([sei_mean], sei_bands))
        index = ['SEI_mean'] + [f'SEI_band_{b}' for b in range(opt.msfa_size ** 2)]
        pd.DataFrame(periodic_avg_dict, index=index).to_csv(
            os.path.join(opt.save_path, f"{type_name}_SEI.csv"), index_label='index')

        del model
        torch.cuda.empty_cache()