import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import os
from PIL import Image
import numpy as np
import zipfile
import io
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fast = True
learning_rate = 0.15 if fast else 0.1
num_epochs = 2 if fast else 5
# num_epochs = 1

class CustomDataset(Dataset):
    def __init__(self, zip_file, data_folder, transform=None):
        self.zip_file = zip_file
        self.data_folder = data_folder
        self.transform = transform
        self.data_paths = self.get_data_paths()

    def get_data_paths(self):
        data_paths = []
        with zipfile.ZipFile(self.zip_file, 'r') as archive:
            for file_info in archive.infolist():
                if file_info.filename.startswith(self.data_folder) and file_info.filename.endswith('.png'):
                    data_paths.append(file_info.filename)
        return data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.zip_file, 'r') as archive:
            data_img_data = archive.read(self.data_paths[idx])
            data_image = Image.open(io.BytesIO(data_img_data))

        if self.transform:
            data_image = self.transform(data_image)

        return data_image

def separate(input_tensor):
    result_tensor = torch.zeros_like(input_tensor, dtype=torch.float32)
    result_tensor[input_tensor < 0.25] = 0
    result_tensor[(input_tensor >= 0.25) & (input_tensor < 0.5)] = 1
    result_tensor[(input_tensor >= 0.5) & (input_tensor < 0.75)] = 2
    result_tensor[input_tensor >= 0.75] = 3

    return result_tensor

def one_hot(seg_map):
    num_classes = 4  # Number of classes
    seg_map = seg_map.squeeze(0)  # Remove batch dimension if present
    one_hot = torch.zeros(num_classes, seg_map.size(0), seg_map.size(1))

    for class_idx in range(num_classes):
        one_hot[class_idx] = (seg_map == class_idx).float()

    return one_hot

size = 128 if fast else 256

transform_X = transforms.Compose([
    transforms.Resize((size, size)),  # Resize the image to the desired size
    transforms.ToTensor(),            # Convert the image to a tensor
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Lambda(lambda x: (x - x.mean()) / x.std())  # Subtract mean and divide by standard deviation
])

transform_Y = transforms.Compose([
    transforms.Resize((size, size)),  # Resize the image to the desired size
    transforms.ToTensor(),            # Convert the image to a tensor
    transforms.Lambda(lambda x: separate(x)),
    transforms.Lambda(one_hot)
])

# Specify the path to your zipped folder
zip_file_path = 'keras_png_slices_data.zip'

X_train = CustomDataset(zip_file=zip_file_path, data_folder=r'keras_png_slices_data/keras_png_slices_train', transform=transform_X)
Y_train = CustomDataset(zip_file=zip_file_path, data_folder=r'keras_png_slices_data/keras_png_slices_seg_train', transform=transform_Y)
X_train_loader = DataLoader(X_train, batch_size=32, shuffle=False)
Y_train_loader = DataLoader(Y_train, batch_size=32, shuffle=False)

X_train = CustomDataset(zip_file=zip_file_path, data_folder=r'keras_png_slices_data/keras_png_slices_train', transform=transform_X)
Y_train = CustomDataset(zip_file=zip_file_path, data_folder=r'keras_png_slices_data/keras_png_slices_seg_train', transform=transform_Y)
X_train_loader = DataLoader(X_train, batch_size=32, shuffle=False)
Y_train_loader = DataLoader(Y_train, batch_size=32, shuffle=False)

X_test = CustomDataset(zip_file=zip_file_path, data_folder=r'keras_png_slices_data/keras_png_slices_test', transform=transform_X)
Y_test = CustomDataset(zip_file=zip_file_path, data_folder=r'keras_png_slices_data/keras_png_slices_seg_test', transform=transform_Y)
X_test_loader = DataLoader(X_test, batch_size=32, shuffle=False)
Y_test_loader = DataLoader(Y_test, batch_size=32, shuffle=False)

import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, initial_size=64, layer_count = 1):
        super(UNet, self).__init__()
        self.size = initial_size

        self.down = nn.ModuleList()
        # Initial down layer
        self.down.append(self._make_layer(in_channels, self.size, self.size, False))
        # Extra down layers
        for i in range(layer_count - 1):
          self.down.append(self._make_layer(self.size, self.size * 2, self.size * 2, False))
        # Middle layer / Initial up layer
        self.mid = self._make_layer(self.size, self.size * 2, self.size, True)

        self.up = nn.ModuleList()
        # Extra up layers
        for i in range(layer_count - 1):
          self.up.append(self._make_layer(self.size * 2, self.size, self.size // 2, True))
        # Output layer
        self.up.append(self._make_layer(self.size * 2, self.size, out_channels, None))

    def _make_layer(self, in_channels, mid_channels, out_channels, up):
        layers = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        if up is None:
            layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1))
        elif up:
            layers.append(nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=4, stride=2, padding=1))
        self.size = out_channels
        return nn.Sequential(*layers)

    def copy(self, toCopy):
        copied = toCopy.clone()
        return nn.MaxPool2d(kernel_size=2, stride=2)(toCopy), copied

    def crop_cat(self, toCrop, toCat):
        cropped = toCrop[:, :, :toCat.size(2), :toCat.size(3)]
        return torch.cat([toCat, cropped], dim=1)

    def forward(self, x):
      copies = []
      result = x
      for layer in self.down:
        result = layer(result)
        result, copy = self.copy(result)
        copies.append(copy)

      result = self.mid(result)

      copies.reverse()
      for i, layer in enumerate(self.up):
        result = layer(self.crop_cat(copies[i], result))

      return result

layer_count = 1 if fast else 2
# Create an instance of the U-Net model
model = UNet(in_channels=1, out_channels=4, layer_count=layer_count)

# Print the model architecture
if device.type == 'cuda':
  print(torch.cuda.get_device_name(0))
  model = model.to(device)

print(sum([param.nelement() for param in model.parameters()]))
print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
total_step = len(X_train_loader)
sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr=learning_rate, step_size_up=15, step_size_down=15, mode="triangular", verbose=False)
sched_linear_3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.005/learning_rate, end_factor=0.005/learning_rate, verbose=False)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30])


model.train()
print("Training")
start = time.time()

for epoch in range(num_epochs):
  for i, (images, seg) in enumerate(zip(X_train_loader, Y_train_loader)):
    images = images.to(device)
    seg = seg.to(device)
    outputs = model(images)
    loss = criterion(outputs, seg)
    # print(outputs)
    # print(seg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
      print(f"Epoch [{epoch + 1} / {num_epochs}], Step [{i + 1} / {total_step} Loss {loss.item()}]")

  scheduler.step()
end = time.time()

print(round((end - start) / 60, 2))

model.eval()
s = False
with torch.no_grad():
    dice_coeff_sum = 0.0
    num_samples = 0
    for i, (images, seg) in enumerate(zip(X_train_loader, Y_train_loader)):
        img = images.to(device)
        seg = seg.to(device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        _, segmax = torch.max(seg, 1)
        output_one_hot = torch.nn.functional.one_hot(predicted, num_classes=4)
        seg_one_hot = torch.nn.functional.one_hot(segmax, num_classes=4)
        if not s:
          vizualize(segmax[0].unsqueeze(0))
          vizualize(predicted[0].unsqueeze(0))
          s = True
        # Calculate Dice coefficient for each image in the batch
        dice_coeff = 2.0 * (output_one_hot * seg_one_hot).sum() / (output_one_hot.sum() + seg_one_hot.sum())

        dice_coeff_sum += dice_coeff.item()
        num_samples += 1

    # Calculate the average Dice coefficient
    average_dice_coeff = dice_coeff_sum / num_samples

    print(f"Average Dice Coefficient: {average_dice_coeff:.4f}")
