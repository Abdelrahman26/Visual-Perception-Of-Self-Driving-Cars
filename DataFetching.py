import os
import torch

# Fetching data from directory folders and store it in lists
data_dir = os.path.join("/kaggle", "input", "cityscapes-image-pairs", "cityscapes_data")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
train_lst = os.listdir(train_dir)
val_lst = os.listdir(val_dir)


# length of data
def length_of_data_lst(data_lst):
    return len(data_lst)


# get image id
def get_img_id(data, idx):
    return data[idx]


def check_device():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    return device