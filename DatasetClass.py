import os
import Helper as pre
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Cityscape_dataset(Dataset):
    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.image_lst = os.listdir(image_dir)
        self.label_model = label_model

    def __len__(self):
        return len(self.image_lst) 

    def __getitem__(self, idx):
        image_name = self.image_lst[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image_colored = Image.open(image_path).convert('RGB')
        cityscape, label = pre.split_image(image_colored)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        cityscape = self.transform(cityscape)
        label_class = torch.tensor(label_class).long()
        return cityscape, label_class

    def split_image(self, img):
        image = np.array(img)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label

    def transform(self, img):
        Transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return Transform(img)
