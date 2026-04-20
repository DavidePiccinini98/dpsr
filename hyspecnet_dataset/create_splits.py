import glob
import numpy as np
import os
import random
import shutil

random.seed(10587)

in_directory = "/media/SSD2/piccinini/hyspecnet-11k/hyspecnet-11k/patches/"

out_directory = "/media/SSD2/piccinini/hyspecnet-11k/hyspecnet-11k/splits/"

train_percent = 0.7
val_percent = 0.2
test_percent = 0.1

if os.path.exists(out_directory):
    shutil.rmtree(out_directory)
os.makedirs(out_directory)
os.makedirs(f"{out_directory}easy/")
os.makedirs(f"{out_directory}hard/")



# Create Split For Train, Validation and Test Set (EASY)
# 
in_patches = glob.glob(f"{in_directory}**/**/*DATA.npy")

in_patches = [("/").join(patch.split("/")[-3:]) for patch in in_patches]

random.shuffle(in_patches)

num_patches = len(in_patches)

train_patches_easy, val_patches_easy, test_patches_easy = np.split(in_patches, [int(num_patches*train_percent), int(num_patches*(train_percent+val_percent))])

train_patches_easy = list(train_patches_easy)
val_patches_easy = list(val_patches_easy)
test_patches_easy = list(test_patches_easy)

train_patches_easy.sort()
val_patches_easy.sort()
test_patches_easy.sort()

np.savetxt(f"{out_directory}easy/train.csv", train_patches_easy, delimiter =", ", fmt ='% s')
np.savetxt(f"{out_directory}easy/val.csv", val_patches_easy, delimiter =", ", fmt ='% s')
np.savetxt(f"{out_directory}easy/test.csv", test_patches_easy, delimiter =", ", fmt ='% s')

#Create Split For Train, Validation and Test Set (HARD)

in_tiles_folders = glob.glob(f"{in_directory}**/")

random.shuffle(in_tiles_folders)

num_tiles = len(in_tiles_folders)

train_tiles_hard, val_tiles_hard, test_tiles_hard = np.split(in_tiles_folders, [int(num_tiles*train_percent), int(num_tiles*(train_percent+val_percent)+1)])

train_patches_hard = []

for tile in train_tiles_hard:
    train_patches_hard.append(glob.glob(f"{tile}**/*DATA.npy"))

train_patches_hard = sum(train_patches_hard, [])
train_patches_hard = [("/").join(patch.split("/")[-3:]) for patch in train_patches_hard]

val_patches_hard = []

for tile in val_tiles_hard:
    val_patches_hard.append(glob.glob(f"{tile}**/*DATA.npy"))

val_patches_hard = sum(val_patches_hard, [])
val_patches_hard = [("/").join(patch.split("/")[-3:]) for patch in val_patches_hard]

test_patches_hard = []

for tile in test_tiles_hard:
    test_patches_hard.append(glob.glob(f"{tile}**/*DATA.npy"))

test_patches_hard = sum(test_patches_hard, [])
test_patches_hard = [("/").join(patch.split("/")[-3:]) for patch in test_patches_hard]

train_patches_hard.sort()
val_patches_hard.sort()
test_patches_hard.sort()

np.savetxt(f"{out_directory}hard/train.csv", train_patches_hard, delimiter =", ", fmt ='% s')
np.savetxt(f"{out_directory}hard/val.csv", val_patches_hard, delimiter =", ", fmt ='% s')
np.savetxt(f"{out_directory}hard/test.csv", test_patches_hard, delimiter =", ", fmt ='% s')