"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import torch.nn.functional as F
import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image

import os

home_dir = os.path.expanduser('~')
test_data_dir = os.path.join(home_dir, 'PyTorch-GAN/data/test_data')
test_image_paths = [os.path.join(test_data_dir, filename) for filename in os.listdir(test_data_dir)]

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

import cv2
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to tensors
    # Add other transformations as needed
])

# Define a custom test dataset class
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, filename) for filename in os.listdir(root)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)  # Load image using OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        return image


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize an empty list to store PSNR values
psnr_values = []
ssim_values = []

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        win_size = 7
        # Calculate SSIM for each generated image
        for j in range(len(imgs_hr)):
        # Calculate SSIM with appropriate win_size
            win_size = min(win_size, min(imgs_hr[j].shape[:2]))
            ssim_value = ssim(
            imgs_hr[j].cpu().numpy().transpose(1, 2, 0),
            gen_hr[j].detach().cpu().numpy().transpose(1, 2, 0),
            multichannel=True,
            win_size=win_size,
            data_range=1.0  # Adjust this value according to your image data range
            )
            

        batch_resized = []
        for img in gen_hr:
            resized_img = F.interpolate(img.unsqueeze(0), size=(imgs_hr.size(2), imgs_hr.size(3)), mode='bilinear', align_corners=False).squeeze(0)
            batch_resized.append(resized_img)

        gen_hr_resized = torch.stack(batch_resized)

        #print("Generated HR image shape:", gen_hr.shape)
        #print("Ground Truth HR image shape:", imgs_hr.shape)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)
        # Compute PSNR
        psnr_value = PSNR(imgs_hr, gen_hr_resized)
        #print("image shape: ", gen_hr_resized.shape )
        print("[PSNR: %f]" % psnr_value.item())
        print(f"[SSIM: {ssim_value}]")

        psnr_values.append(psnr_value.item())
        ssim_values.append(ssim_value)
    torch.save(generator, "saved_models/generator_model1.pth")
    torch.save(discriminator, "saved_models/discriminator_model1.pth")
    #if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
    
#import saved model
#generator = torch.load("saved_models/generator_model.pth")
#discriminator = torch.load("saved_models/discriminator_model.pth")
""""
if cuda:
    generator = generator.cuda()
else:
    generator = generator.cpu()

low_res_height = 256
low_res_width = 256
#evaluation mode set
#generator.eval()
#discriminator.eval()
test_image = Image.open("000278.jpg")
preprocess = transforms.Compose([
    transforms.Resize((low_res_height, low_res_width)),
    transforms.ToTensor(),  # Convert PIL Image to tensor
    # Add other preprocessing steps if needed
])
test_image_tensor = preprocess(test_image).unsqueeze(0)  # Add batch dimension

# Load the model
generator = torch.load("saved_models/generator_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
generator.eval()  # Set the model to evaluation mode

with torch.no_grad():  
    test_image_tensor = test_image_tensor.to(device)
    generated_image = generator(test_image_tensor)

#final_output = postprocess(output_data)
save_image(test_image, "generated_image.jpg")
"""""

# Path to the test data directory
#test_data_dir = "~/PyTorch-GAN/data/test_data"

# Create a custom test dataset instance
test_dataset = TestDataset(test_data_dir, transform=transform)

# Create a data loader for the test dataset
batch_size_test = 4
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# Iterate over the test loader to access images
for batch_idx, images in enumerate(test_loader):
    # Process images as needed
    print(f"Batch {batch_idx}: {images.shape}")  # Example: Print shape of images tensor

# Create the test dataset
#test_dataset = TestDataset("/test_data", transform=transform)

# Create a data loader for the test dataset
#test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
psnr_test_values = []
ssim_test_values = []
# Load the generator model
#generator = torch.load("saved_models/generator_model.pth",map_location=device)

# Set the device for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
generator = torch.load("saved_models/generator_model.pth",map_location=device)
generator.eval()

# Perform inference on the test dataset
with torch.no_grad():
    for batch_idx, images in enumerate(test_loader):
        images = images.to(device)
        generated_images = generator(images)
        save_image(generated_images, f"generated_images_batch_{batch_idx}.png")
        # Process the generated images as needed

for image_path in test_image_paths:
    # Load the test image
    test_image = Image.open(image_path).convert("RGB")

    # Preprocess the test image
    test_image_tensor = transform(test_image).unsqueeze(0).to(device)  # Add batch dimension
    # Perform inference to generate super-resolved image
    with torch.no_grad():
        generated_image = generator(test_image_tensor)

    generated_image_np = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Resize the generated image to match the dimensions of the original high-resolution image
    generated_image_resized = cv2.resize(generated_image_np, (test_image.width, test_image.height))

    # Convert the original high-resolution image to numpy array
    test_image_np = np.array(test_image)

    # Calculate PSNR between the original and resized generated images
    psnr = peak_signal_noise_ratio(test_image_np, generated_image_resized)
    psnr_test_values.append(psnr)

    # Calculate SSIM for the pair of images
    #ssim_test_value = ssim(test_image_np, generated_image_resized, multichannel=True, win_size=(min(test_image_np.shape[0], test_image_np.shape[1], 7),))
    ssim_test_value = ssim(
    test_image_np,
    generated_image_resized,
    multichannel=True,
    win_size=win_size,
    data_range=test_image_np.max() - test_image_np.min()  # Calculate data range based on the image
    )
    ssim_test_values.append(ssim_test_value)

# Calculate the average SSIM value
#average_ssim = sum(ssim_test_values) / len(ssim_test_values)

# Print or use the PSNR values as needed
print("PSNR values for test images:", psnr_test_values)
print("SSIM values for test images:", ssim_test_values)
# Plotting the PSNR values
plt.plot(range(len(psnr_test_values)), psnr_test_values, label='PSNR', marker='o', linestyle='-')
plt.xlabel('Test Image Index')
plt.ylabel('PSNR')
plt.title('PSNRs for Test Images')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("psnrtest")

# Assuming ssim_values is a list or array containing SSIM values for each epoch
#epochs = range(1, len(ssim_values) + 1)

# Plotting SSIM values against epochs
#plt1.plot(range(1, len(ssim_values) + 1), ssim_values, label='SSIM', linestyle='-')
#plt1.title('SSIM Value vs Epochs')
#plt1.xlabel('Epochs')
#plt1.ylabel('SSIM Value')
#plt1.legend()
#plt1.grid(True)
#plt1.show()
#plt1.savefig("ssim")