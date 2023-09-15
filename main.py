# Import Libraries
import gc
import glob
# import json
import os
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

# Load Config files
# path = "/content/drive/My Drive/YAAR/yaar_"
# config_path = os.path.join(path, "config.json")
# with open(config_path, "r") as f:
#     config = json.load(f)

# print("The Configuration Variables are:")
# print(config)

# Define Config variables
image_size = (400, 640)
data_path = "images/encoder_data/"

batch_size = 4
learning_rate = 0.0001
weight_decay = 0  # config["weight_decay"]
epochs = 25

print("\n____________________________________________________\n")
print("\nLoading Dataset into DataLoader...")

# Get All Images

all_imgs = glob.glob(data_path + "*")
shuffle(all_imgs)
dbg_images = glob.glob("dbg_images/as_input.png")


# Train Images
train_imgs = all_imgs[:2000]
test_imgs = all_imgs[2000:2100]


# DataLoader Function
class imagePrep(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        super().__init__()
        self.paths = images
        self.len = len(self.paths)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)
        image = self.transform(image)
        if "Cat" in path:
            label = 0
        else:
            label = 1

        return (image, label)


# Dataset Transformation Function
dataset_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
    ]
)

# Apply Transformations to Data
Tr_DL = torch.utils.data.DataLoader(
    imagePrep(train_imgs, dataset_transform), batch_size=batch_size
)
Ts_DL = torch.utils.data.DataLoader(
    imagePrep(test_imgs, dataset_transform), batch_size=batch_size
)
s_DL = torch.utils.data.DataLoader(
    imagePrep(dbg_images, dataset_transform), batch_size=1
)
for i in s_DL:
    dbg_image = i[0]
print(dbg_image)

# Open one image
print("\nTest Open One Image")
plt.imshow(Image.open(all_imgs[5]))

print("\nDataLoader Set!")
print("\n____________________________________________________\n")

print("\nBuilding Convolutional AutoEncoder Network Model...")


# Define Convolutional AutoEncoder Network
class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, stride=1, padding=1),  #
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(64, 3, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded

    def save(self, path):
        np.savez(
            path + ".npz",
            **{
                "0.0.0.weight": self.encoder[0].weight.cpu().detach().numpy(),
                "0.0.1.bias": self.encoder[0].bias.cpu().detach().numpy(),
                "0.1.0.weight": self.encoder[3].weight.cpu().detach().numpy(),
                "0.1.1.bias": self.encoder[3].bias.cpu().detach().numpy(),
                "0.2.0.weight": self.encoder[6].weight.cpu().detach().numpy(),
                "0.2.1.bias": self.encoder[6].bias.cpu().detach().numpy(),
                "1.0.1.weight": self.decoder[1].weight.cpu().detach().numpy(),
                "1.0.2.bias": self.decoder[1].bias.cpu().detach().numpy(),
                "1.1.1.weight": self.decoder[4].weight.cpu().detach().numpy(),
                "1.1.2.bias": self.decoder[4].bias.cpu().detach().numpy(),
                "1.2.1.weight": self.decoder[7].weight.cpu().detach().numpy(),
                "1.2.2.bias": self.decoder[7].bias.cpu().detach().numpy(),
            }
        )
        torch.save(self, path)


print("\nConvolutional AutoEncoder Network Model Set!")

print("\n____________________________________________________\n")

# defining the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining the model
convAE_model = ConvAutoencoder().to(device)
np.save( "dbg_images/npin", dbg_image.numpy())
dbg_image = dbg_image.to(device)
try:
    convAE_model = torch.load("ai-file/pysave").to(device)
except Exception:
    print("Could not load")
    pass


# defining the optimizer
optimizer = torch.optim.Adam(
    convAE_model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

# defining the loss function
loss_function = torch.nn.MSELoss().to(device)

print(convAE_model)
print("____________________________________________________\n")

print("\nTraining the Convolutional AutoEncoder Model on Training Data...")

# Training of Model

# losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    done = 0
    for X, y in Tr_DL:
        img = X.to(device)
        img = torch.autograd.Variable(img)

        mid = convAE_model.encoder(img)
        recon = convAE_model.decoder(mid)

        loss = loss_function(recon, img)

        # Backward Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss
        # print("-", end="", flush=True)
        print(done, "/", len(Tr_DL), end="\r", flush=True)

        if done % 100 == 0:
            print(img.shape, mid.shape, recon.shape)
            # print(img[0, 0, 69:79, 470:480])
            torchvision.utils.save_image(
                img[0], "dbg_images/pyinput{}.png".format(done)
            )
            torchvision.utils.save_image(
                recon[0], "dbg_images/pyoutput{}.png".format(done)
            )
        done += 1

    epoch_loss = epoch_loss / len(Tr_DL)
    # losses.append(epoch_loss)
    convAE_model.save("ai-file/pymodel")
    r = convAE_model(dbg_image)
    np.save("dbg_images/outp", r.cpu().detach().numpy())
    torchvision.utils.save_image(r, "dbg_images/encoded_py.png")
    gc.collect()

    print("\nEpoch: {} | Loss: {:.4f}".format(epoch + 1, epoch_loss))
    if epoch_loss < 0.0001:
        break

print("\n____________________________________________________\n")

fig = plt.figure(figsize=(12, 5))

# plt.plot(losses, "-r", label="Training loss")
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.title("Convolutional AutoEncoder Training Loss Vs Epochs", fontsize=15)
plt.show()

print("\n____________________________________________________\n")

print("PRINTING ORIGINAL IMAGES THAT TRAINED THE MODEL AND THEIR RECONSTRUCTIONS ...")

# Print Some Reconstructions
plt.figure(figsize=(23, 8))

start = 4
n_images = 5

for i in range(n_images):
    plt.subplot(1, n_images, i + 1)
    plt.imshow(X[start + i + 1][0], cmap="gray")
    plt.title("Training Image " + str(i + 1), fontsize=15)
    plt.axis("off")

plt.figure(figsize=(23, 8))

for i in range(n_images):
    plt.subplot(1, n_images, i + 1)
    pic = recon.cpu().data
    plt.imshow(pic[start + i + 1][0], cmap="gray")
    plt.title("Reconstructed Image " + str(i + 1), fontsize=15)
    plt.axis("off")

print("\n____________________________________________________\n")

print("\n____________________________________________________\n")

# Reconstruct Images by passing Test images on Trained Model
with torch.no_grad():
    for Ts_X, Ts_y in Ts_DL:
        Ts_X = Ts_X.to(device)
        Ts_y = Ts_y.to(device)

        Ts_recon = convAE_model(Ts_X)


print("PRINTING TEST IMAGES AND THEIR RECONSTRUCTIONS ...")
print("\n____________________________________________________\n")

# Print Some Reconstructions
plt.figure(figsize=(23, 8))

start = 4
n_images = 5

for i in range(n_images):
    plt.subplot(1, n_images, i + 1)
    pic = Ts_X.cpu().data
    plt.imshow(pic[start + i + 1][0], cmap="gray")
    plt.title("Test Image " + str(i + 1), fontsize=15)
    plt.axis("off")

plt.figure(figsize=(23, 8))

for i in range(n_images):
    plt.subplot(1, n_images, i + 1)
    pic = Ts_recon.cpu().data
    plt.imshow(pic[start + i + 1][0], cmap="gray")
    plt.title("Reconstructed Image " + str(i + 1), fontsize=15)
    plt.axis("off")


print("\n____________________________________________________\n")
