# HySpecNet Tools
This repository contains tools for processing the HySpecNet-11k benchmark dataset. For downloading the dataset and a detailed explanation, please visit the HySpecNet website at [https://hyspecnet.rsim.berlin/](https://hyspecnet.rsim.berlin/). This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) by [Martin Hermann Paul Fuchs](https://rsim.berlin/team/members/martin-hermann-paul-fuchs) and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

If you use this code, please cite our paper given below:

> M. H. P. Fuchs and B. Demir, "[HySpecNet-11k: a Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods,](https://arxiv.org/abs/2306.00385)" IEEE International Geoscience and Remote Sensing Symposium, Pasadena, CA, USA, 2023, pp. 1779-1782, doi: 10.1109/IGARSS52108.2023.10283385.
```
@INPROCEEDINGS{10283385,
  author={Fuchs, Martin Hermann Paul and Demir, Begüm},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={HySpecNet-11k: a Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods}, 
  year={2023},
  volume={},
  number={},
  pages={1779-1782},
  doi={10.1109/IGARSS52108.2023.10283385}}
```

## Setup
The code in this repository is tested with `Ubuntu 22.04 LTS` and `Python 3.10.6`.

### Dependencies
All dependencies are listed in the [`requirements.txt`](requirements.txt) and can be installed via the following command:
```
pip install -r requirements.txt
```

### Download
Follow the instructions on [https://hyspecnet.rsim.berlin](https://hyspecnet.rsim.berlin) to download HySpecNet-11k.

The folder structure should be as follows:
```
┗ 📂 hyspecnet-11k/
  ┣ 📂 patches/
  ┃ ┣ 📂 tile_001/
  ┃ ┃ ┣ 📂 tile_001-patch_01/
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-DATA.npy
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_PIXELMASK.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CIRRUS.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CLASSES.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CLOUD.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CLOUDSHADOW.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_HAZE.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_SNOW.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_TESTFLAGS.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_SWIR.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_VNIR.TIF
  ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-SPECTRAL_IMAGE.TIF
  ┃ ┃ ┃ ┗ 📜 tile_001-patch_01-THUMBNAIL.jpg
  ┃ ┃ ┣ 📂 tile_001-patch_02/
  ┃ ┃ ┃ ┗ 📜 ...
  ┃ ┃ ┗ 📂 ...
  ┃ ┣ 📂 tile_002/
  ┃ ┃ ┗ 📂 ...
  ┃ ┗ 📂 ...
  ┗ 📂 splits/
  ┣ 📂 easy/
  ┃ ┣ 📜 test.csv
  ┃ ┣ 📜 train.csv
  ┃ ┗ 📜 val.csv
  ┣ 📂 hard/
  ┃ ┣ 📜 test.csv
  ┃ ┣ 📜 train.csv
  ┃ ┗ 📜 val.csv
  ┗ 📂 ...
```

## Usage

### Create Numpy Files
To generate the preprocessed `*-DATA.npy` files, run the [tif_to_npy.ipynb](tif_to_npy.ipynb) notebook.

### Create Dataset Split
[create_splits.ipynb](create_splits.ipynb) shows how we created the dataset splits.

## Authors
**Martin Hermann Paul Fuchs**
https://rsim.berlin/team/members/martin-hermann-paul-fuchs

## License
The code in this repository is licensed under the **MIT License**:
```
MIT License

Copyright (c) 2023 Martin Hermann Paul Fuchs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
