import os
import glob
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import random


class GenericFolderDataset(data.Dataset):
    def __init__(self, image_dir, msfa_size, msfa_pattern, norm_flag=False, patch_size=120, augment=False,
                 type='train', input_transform=None, target_transform=None):
        super(GenericFolderDataset, self).__init__()
        self.image_dir = image_dir
        # Ordine deterministico delle cartelle
        self.sample_dirs = sorted([os.path.join(image_dir, d) for d in os.listdir(image_dir)
                                   if os.path.isdir(os.path.join(image_dir, d))])

        self.msfa_size = msfa_size
        self.msfa_pattern = msfa_pattern
        self.patch_size = patch_size
        # patch_size=None -> immagine intera (validazione)
        self.crop_size = None if patch_size is None else patch_size - (patch_size % msfa_size)
        self.type = type
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment = augment
        self.norm_flag = norm_flag

    def __getitem__(self, index):
        real_index = index % len(self.sample_dirs)
        sample_dir = self.sample_dirs[real_index]
        orig_dir = os.path.join(sample_dir, 'original_bands')
        sparse_dir = os.path.join(sample_dir, 'sparse_matrices')

        # 1. Apriamo un'immagine di riferimento per ricavare le dimensioni h, w senza caricarla interamente
        sample_file = os.path.join(orig_dir, f'original_band_{self.msfa_pattern[0]}.tiff')
        with Image.open(sample_file) as img_ref:
            w, h = img_ref.size

        # 2. Calcoliamo le coordinate del crop in anticipo
        if self.crop_size is None:
            # Immagine intera, ritagliata al multiplo di msfa_size
            cw = w - (w % self.msfa_size)
            ch = h - (h % self.msfa_size)
            x, y = 0, 0
        elif self.type == 'train':
            cw = ch = self.crop_size
            x = random.randint(0, max(0, (w - cw) // self.msfa_size)) * self.msfa_size
            y = random.randint(0, max(0, (h - ch) // self.msfa_size)) * self.msfa_size
        else:
            cw = ch = self.crop_size
            x, y = 0, 0  # Offset deterministico e fisso per test/validazione

        box = (x, y, x + cw, y + ch)

        target_list = []
        sparse_list = []

        # 3. Carichiamo e ritagliamo SOLO la porzione di interesse direttamente tramite PIL
        for band_idx in self.msfa_pattern:
            orig_file = os.path.join(orig_dir, f'original_band_{band_idx}.tiff')
            sparse_file = os.path.join(sparse_dir, f'sparse_matrix_{band_idx}.tiff')

            with Image.open(orig_file) as img:
                cropped_img = img.crop(box)
                target_list.append(np.array(cropped_img, dtype=np.float32))

            with Image.open(sparse_file) as img:
                cropped_img = img.crop(box)
                sparse_list.append(np.array(cropped_img, dtype=np.float32))

        target = np.stack(target_list, axis=-1)
        input_image = np.stack(sparse_list, axis=-1)

        mosaic_file = glob.glob(os.path.join(sample_dir, '*Mosaic*.tif*'))[0]
        with Image.open(mosaic_file) as img:
            cropped_mosaic = img.crop(box)
            raw = np.array(cropped_mosaic, dtype=np.float32)

        # 4. Data Augmentation (solo se abilitata)
        if self.augment and self.type == 'train':
            if np.random.uniform() < 0.5:
                target = np.ascontiguousarray(np.fliplr(target))
                input_image = np.ascontiguousarray(np.fliplr(input_image))
                raw = np.ascontiguousarray(np.fliplr(raw))
            if np.random.uniform() < 0.5:
                target = np.ascontiguousarray(np.flipud(target))
                input_image = np.ascontiguousarray(np.flipud(input_image))
                raw = np.ascontiguousarray(np.flipud(raw))
            k_rot = np.random.randint(0, 4)
            target = np.ascontiguousarray(np.rot90(target, k=k_rot))
            input_image = np.ascontiguousarray(np.rot90(input_image, k=k_rot))
            raw = np.ascontiguousarray(np.rot90(raw, k=k_rot))

        if self.norm_flag:
            raw /= 255.0
            input_image /= 255.0
            target /= 255.0

        raw = torch.from_numpy(raw).unsqueeze(0)
        input_image = torch.from_numpy(input_image).permute(2, 0, 1)
        target = torch.from_numpy(target).permute(2, 0, 1)

        return raw, input_image, target

    def __len__(self):
        # 5 crop casuali per immagine in training, 1 sola passata in validazione
        return len(self.sample_dirs) * (5 if self.type == 'train' else 1)