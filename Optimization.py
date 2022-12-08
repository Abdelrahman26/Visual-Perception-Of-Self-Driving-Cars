import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from DatasetClass import Cityscape_dataset
from DataFetching import train_dir, check_device
from Helper import label_model
from UNet import UNET

device = check_device()
batch_size = 16
epochs = 10
lr = 0.001

dataset = Cityscape_dataset(train_dir, label_model)
data_loader = DataLoader(dataset, batch_size=batch_size)

model = UNET(in_channels=3, out_channels=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

step_losses = []
epoch_losses = []

for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
    print(epoch_loss)
    epoch_losses.append(epoch_loss/len(data_loader))


def visulaizeLoss():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[1].plot(epoch_losses)

def save_parameters(PATH):
    torch.save(model.state_dict(), PATH)

def load_parameters(model, PATH):
    state_dict = torch.load(PATH)
    model.load_state_dict(state_dict)



