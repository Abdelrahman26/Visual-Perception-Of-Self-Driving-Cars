import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from DatasetClass import Cityscape_dataset
from DataFetching import val_dir, check_device
from UNet import UNET
from Helper import label_model
import numpy as np
import torch


test_batch_size = 8
dataset = Cityscape_dataset(val_dir, label_model)
data_loader = DataLoader(dataset, batch_size=test_batch_size)


# get original channels(+ mean, * std)
inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])


# install model parameters (weights + biases)
device = check_device()
model_ = UNET(in_channels=3, out_channels=10).to(device)
model_.load_state_dict(torch.load("model_path"))


X, Y = next(iter(data_loader))


Y_pred = model_(X)
Y_pred = torch.argmax(Y_pred, dim=1)


iou_scores = []
for i in range(test_batch_size):
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()

    # IOU score
    intersection = np.logical_and(label_class, label_class_predicted)
    union = np.logical_or(label_class, label_class_predicted)
    iou_score = np.sum(intersection) / np.sum(union)
    iou_scores.append(iou_score)




def visualize():
    fig, axes = plt.subplots(test_batch_size, 3, figsize=(3 * 5, test_batch_size * 5))
    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")