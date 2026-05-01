import os
import re
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from monai import transforms as monai_transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm
import random


def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f % frames == 0:
        return t[:, :-(frames - 1)]
    if f % frames == 1:
        return t
    else:
        return t[:, :-((f % frames) - 1)]


class MCT2SMRDataset(Dataset):
    def __init__(self, args, data_folder, xlsx_file, tokenizer, min_slices=20, resize_dim=500, num_frames=2,
                 force_num_frames=True, mode='train'):
        self.data_folder = data_folder
        self.mask_paths = args.mask_paths
        self.phase_names = args.phase_name
        self.mask_names = args.mask_name
        self.a_min = args.a_min
        self.a_max = args.a_max
        self.roi_x = args.roi_x
        self.roi_y = args.roi_y
        self.roi_z = args.roi_z
        self.b_min = args.b_min
        self.b_max = args.b_max
        self.min_slices = min_slices
        self.xlsx_file = xlsx_file
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        random.seed(1234)
        self.samples = self.prepare_samples()
        # self.samples = self.samples[:1]
        self.samples = random.sample(self.samples, int(len(self.samples)))
        if mode == 'val':
            self.samples = self.samples[:3]

        self.paths = []

        self.transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor()
        ])

        self.seg_transform = monai_transforms.Compose(
            [
                monai_transforms.LoadImaged(keys=["mask"]),
                monai_transforms.AddChanneld(keys=["mask"]),
                monai_transforms.Orientationd(keys=["mask"], axcodes="RAS"),
                monai_transforms.Resized(
                    keys=["mask"],
                    spatial_size=(self.roi_x, self.roi_y, self.roi_z),
                    mode=["nearest"],
                ),
                monai_transforms.ToTensord(keys=["mask"]),
            ]
        )

        self.seg_nii_to_tesnsor = partial(self.seg_nii_img_to_tesnsor, seg_transform=self.seg_transform)
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform=self.transform)
        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames)


    def prepare_samples(self):
        df = pd.read_excel(self.xlsx_file)
        sample_names = df["图像名称"]
        self.paths = [os.path.join(self.data_folder, sample_name + ".nii.gz") for sample_name in sample_names]
        samples = []
        for index, row in df.iterrows():

            row_text = (
                f"{row['报告表现']}。"
            )

            b1 = f"肿块评估：" + re.search(r"肿块评估：\s*(.*?)\s*胰胆管评估：", row_text).group(1).strip()
            b2 = f"胰胆管评估：" + re.search(r"胰胆管评估：\n(.*?)(?=\n\S+评估：|$)", row_text, re.DOTALL).group(1).strip()
            b3 = f"动脉评估：\n" + re.search(r"动脉评估：\n(.*?)(?=\n\S+评估：|$)", row_text, re.DOTALL).group(1).strip()
            b4 = f"静脉评估：\n" + re.search(r"静脉评估：\n(.*?)(?=\n\S+评估：|$)", row_text, re.DOTALL).group(1).strip()
            b5 = "其它评估：\n" + re.search(r"其它评估：\n(.*?)(?=\n\S+描述：|$)", row_text, re.DOTALL).group(1).strip()
            b6 = f"其它放射学描述：" + re.search(r"其它放射学描述：(.*?)(?=\n|$)", row_text).group(1).strip()


            block_text = [b1, b2, b3, b4, b5, b6]
            img_paths = []
            mask_paths = []
            for phase_name, mask_name in zip(self.phase_names, self.mask_names) :
                imgpath = os.path.join(self.data_folder, f"{phase_name}/{row['图像名称']}.nii.gz")
                maskpath = os.path.join(self.mask_paths, f"{mask_name}/{row['图像名称']}.nii.gz")

                img_paths.append(imgpath)
                mask_paths.append(maskpath)

            samples.append((img_paths, mask_paths, row_text, block_text))
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        img_data = nib.load(path).get_fdata()

        hu_min, hu_max = self.a_min, self.a_max
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (img_data / 1000).astype(np.float32)

        tensor = torch.tensor(img_data)

        target_shape = (self.roi_x, self.roi_y, self.roi_z)

        h, w, d = tensor.shape

        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (
        pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor[0]


    def seg_nii_img_to_tesnsor(self, path, seg_transform):

        data_dict = {
            "mask": path
        }
        trans_data = seg_transform(data_dict)
        seg_tensor = trans_data["mask"].permute(0, 3, 1, 2)

        return seg_tensor


    def __getitem__(self, index):
        img_paths, mask_paths, text, block_text = self.samples[index]
        img_id = img_paths[0].split("/")[-1]
        tensor = [self.nii_to_tensor(path) for path in img_paths]

        seg_tensor = [self.seg_nii_to_tesnsor(path) for path in mask_paths]
        # print(f"row text: {text}")

        tensor = torch.cat(tensor)
        seg_tensor = torch.cat(seg_tensor)

        ids = self.tokenizer(text)[:self.max_seq_length]
        block_ids = [self.tokenizer(txt) for txt in block_text]
        blocktext_len_list = [len(block_len) for block_len in block_ids]

        mask = [1] * len(ids)
        seq_lenght = len(ids)

        sample = (img_id, tensor, seg_tensor, ids, mask, seq_lenght, block_ids, blocktext_len_list)
        return sample