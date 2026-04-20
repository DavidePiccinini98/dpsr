import csv
import os

import numpy as np
import torch

from torch.utils.data import Dataset

#def data_augmentation(label, mode=0):
#    if mode == 0:
#        # original
#        return label.copy()
#    elif mode == 1:
#        # flip up and down
#        return np.flipud(label).copy()
#    elif mode == 2:
#        # rotate counterwise 90 degree
#        rotated = np.rot90(label, k=1, axes=(1, 2))
#        return rotated.copy()
#    elif mode == 3:
#        # rotate 90 degree and flip up and down
#        rotated = np.rot90(label,  k=1, axes=(1, 2))
#        return np.flipud(rotated).copy()
#    elif mode == 4:
#        # rotate 180 degree
#        return np.rot90(label, k=2, axes=(1, 2)).copy()
#    elif mode == 5:
#        # rotate 180 degree and flip
#        rotated = np.rot90(label, k=2, axes=(1, 2)).copy()
#        return np.flipud(rotated).copy()
#    elif mode == 6:
#        # rotate 270 degree
#        return np.rot90(label, k=3, axes=(1, 2)).copy()
#    elif mode == 7:
#        # rotate 270 degree and flip
#        return np.flipud(np.rot90(label, k=3, axes=(1, 2))).copy()
def data_augmentation(label, mode=0):
    if mode == 0:
        # original
        return label.copy()
    elif mode == 1:
        # flip up and down
        return np.flipud(label).copy()
    elif mode == 2:
        # rotate counterwise 90 degree
        rotated = np.rot90(label, k=1, axes=(1, 2))
        return rotated.copy()
    elif mode == 3:
        # rotate 90 degree and flip up and down
        rotated = np.rot90(label,  k=1, axes=(1, 2))
        return np.flipud(rotated).copy()
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(label, k=2, axes=(1, 2)).copy()
    elif mode == 5:
        # rotate 180 degree and flip
        rotated = np.rot90(label, k=2, axes=(1, 2)).copy()
        return np.flipud(rotated).copy()
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(label, k=3, axes=(1, 2)).copy()
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(label, k=3, axes=(1, 2))).copy()
class HySpecNet11k(Dataset):
    """
    Dataset:
        HySpecNet-11k
    Authors:
        Martin Hermann Paul Fuchs
        Begüm Demir
    Related Paper:
        HySpecNet-11k: A Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods
        https://arxiv.org/abs/2306.00385
    Cite: TODO
        @misc{fuchs2023hyspecnet11k,
            title={HySpecNet-11k: A Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods}, 
            author={Martin Hermann Paul Fuchs and Begüm Demir},
            year={2023},
            eprint={2306.00385},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }

    Folder Structure:
        - root_dir/
            - patches/
                - tile_001/
                    - tile_001-patch_01/
                        - tile_001-patch_01-DATA.npy
                        - tile_001-patch_01-QL_PIXELMASK.TIF
                        - tile_001-patch_01-QL_QUALITY_CIRRUS.TIF
                        - tile_001-patch_01-QL_QUALITY_CLASSES.TIF
                        - tile_001-patch_01-QL_QUALITY_CLOUD.TIF
                        - tile_001-patch_01-QL_QUALITY_CLOUDSHADOW.TIF
                        - tile_001-patch_01-QL_QUALITY_HAZE.TIF
                        - tile_001-patch_01-QL_QUALITY_SNOW.TIF
                        - tile_001-patch_01-QL_QUALITY_TESTFLAGS.TIF
                        - tile_001-patch_01-QL_SWIR.TIF
                        - tile_001-patch_01-QL_VNIR.TIF
                        - tile_001-patch_01-SPECTRAL_IMAGE.TIF
                        - tile_001-patch_01-THUMBNAIL.jpg
                    - tile_001-patch_02/
                        - ...
                    - ...
                - tile_002/
                    - ...
                - ...
            - splits/
                - easy/
                    - test.csv
                    - train.csv
                    - val.csv
                - hard/
                    - test.csv
                    - train.csv
                    - val.csv
                - ...
            - ...
    """
####    def __init__(self, root_dir, mode="easy", split="train", transform=None):
####        self.root_dir = root_dir
####
####        self.csv_path = os.path.join(self.root_dir, "splits", mode, f"{split}.csv")
####        with open(self.csv_path, newline='') as f:
####            csv_reader = csv.reader(f)
####            csv_data = list(csv_reader)
####            self.npy_paths = sum(csv_data, [])
####        self.npy_paths = [os.path.join(self.root_dir, "patches", x) for x in self.npy_paths]
####
####        self.transform = transform
####
####    def __len__(self):
####        return len(self.npy_paths)
####
####    def __getitem__(self, index):
####        # get full numpy path
####        npy_path = self.npy_paths[index]
####        # read numpy data
####        img = np.load(npy_path)
####        # convert numpy array to pytorch tensor
####        img = torch.from_numpy(img)
####        # apply transformations
####        if self.transform:
####            img = self.transform(img)
####        return img
    
    def __init__(self, root_dir, mode="easy", split="train", transform=None, augment= False):
            self.root_dir = root_dir
    
            self.csv_path = os.path.join(self.root_dir, "splits", mode, f"{split}.csv")
            with open(self.csv_path, newline='') as f:
                csv_reader = csv.reader(f)
                csv_data = list(csv_reader)
                self.npy_paths = sum(csv_data, [])
            self.npy_paths = [os.path.join(self.root_dir, "patches", x) for x in self.npy_paths]
    
            self.transform = transform
            self.augment = augment 
            
            if self.augment:
                self.factor = 8
            else:
                self.factor = 1
    
    def __len__(self):
        
        return len(self.npy_paths)  * self.factor 

    def __getitem__(self, index):
        if self.augment:
            file_index = index // self.factor
            aug_num = index % self.factor  # 0-7
        else:
            file_index = index
            aug_num = 0  # No augmentation

        # get full numpy path
        npy_path = self.npy_paths[file_index]
        # read numpy data
        img = np.load(npy_path)
        
        img = data_augmentation(img, mode=aug_num)
        # apply transformations
#        if self.transform:
#            img = data_augmentation(img, mode=aug_num)
#            img = self.transform(img)
        # convert numpy array to pytorch tensor
        fname = os.path.basename(npy_path) 
        img = torch.from_numpy(img.copy())
        return img#, fname 