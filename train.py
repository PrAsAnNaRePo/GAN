import os
import cv2
import numpy as np
from turtle import forward
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter


IMG_DIR = 'D:/dataset/C&D/training_set/training_set'
IMG_SIZE = (300, 300)
Z_DIM = (1, 32)
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 3e-4

fixed_noise = torch.rand((BATCH_SIZE, 1, Z_DIM[0], Z_DIM[1]))

class Data(Dataset):
    def __init__(self, img_dir, img_size):
        super().__init__()
        self.img_dir = img_dir
        self.img_size = img_size
        self.images_path = os.listdir(self.img_dir)
    def __len__(self):
        return len(self.images_path)
    def __getitem__(self, index):
        img = np.array(Image.open(f'{self.img_dir}/{self.images_path[index]}'))
        img = cv2.resize(img, self.img_size)
        img = torch.tensor(img.reshape(3, 300, 300), dtype=torch.float32)
        return img, 0.0

data = Data(IMG_DIR, IMG_SIZE)

loader = DataLoader(data, BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True)


class ConvBlock(nn.Module):
    def __init__(self, in_f, out_f, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_f, out_f, bias=False, **kwargs),
            nn.BatchNorm2d(out_f),
            nn.MaxPool2d((2, 2)),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.block(x)


class Upsample(nn.Module):
    def __init__(self, size, in_f, out_f, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(size),
            nn.Conv2d(in_f, out_f, bias=False, **kwargs),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.gen = nn.Sequential(
            Upsample((10, 10), z_dim[0], 32, kernel_size=3, padding=1),
            Upsample((50, 50), 32, 64, kernel_size=3, padding=1),
            Upsample((70, 70), 64, 64, kernel_size=3, padding=1),
            Upsample((120, 120), 64, 64, kernel_size=3, padding=1),
            Upsample((140, 140), 64, 32, kernel_size=3, padding=1),
            Upsample((150, 150), 32, 64, kernel_size=3, padding=1),
            Upsample((180, 180), 64, 64, kernel_size=3, padding=1),
            Upsample((200, 200), 64, 64, kernel_size=3, padding=1),
            Upsample((250, 250), 64, 32, kernel_size=3, padding=1),
            Upsample((270, 270), 32, 64, kernel_size=3, padding=1),
            Upsample((290, 290), 64, 32, kernel_size=3, padding=1),
            Upsample(img_size, 32, 3, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.disc = nn.Sequential(
            ConvBlock(3, 32, kernel_size=4, stride=2),
            ConvBlock(32, 64, kernel_size=4),
            ConvBlock(64, 128, kernel_size=3),
            ConvBlock(128, 32, kernel_size=3),
            nn.Flatten(),
            nn.Linear(32*7*7, 100),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.disc(x)

generator = Generator(Z_DIM, IMG_SIZE).to('cuda')
discriminator = Discriminator(IMG_SIZE).to('cuda')
criterion = nn.BCELoss()
optim_gen = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optim_disc = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

writer_lossG = SummaryWriter(f'logs/lossG')
writer_lossD = SummaryWriter(f'logs/lossD')
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

torch.backends.cudnn.benchmark = True

def train(gen:Generator, disc:Discriminator, epochs:int, loss, gen_optim, disc_optim, loader:DataLoader):
    step = 0
    for epoch in range(1, epochs+1):
        for B, (real, _) in enumerate(loader):
            real = real.to('cuda')
            noise = torch.rand((BATCH_SIZE, 1, 1, 128), device='cuda')
            fake = gen(noise)

            disc_real =  disc(real)
            lossD_real = loss(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake)
            lossD_fake = loss(disc_fake, torch.zeros_like(disc_fake))

            lossD = (lossD_real + lossD_fake) / 2
            disc_optim.zero_grad()
            lossD.backward(retain_graph=True)
            disc_optim.step()

            output = disc(fake)
            lossG = loss(output, torch.ones_like(output))
            gen_optim.zero_grad()
            lossG.backward()
            gen_optim.step()


            if B == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {B}/{len(loader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

            with torch.no_grad():
                fake = gen(fixed_noise.to('cuda'))
                data = real
                img_grid_fake = torchvision.utils.make_grid(fake)
                img_grid_real = torchvision.utils.make_grid(data)
                writer_lossG.add_scalar('lossG', lossG, global_step=step)
                writer_lossD.add_scalar('lossD', lossD, global_step=step)
                writer_fake.add_image(
                    "C&D Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "C&D Real Images", img_grid_real, global_step=step
                )
                step += 1
        # print(f'[{epoch}/{epochs}]=====>GanLoss:{lossG.item()}=====>DiscLoss:{lossD.item()}')



if __name__ == '__main__':
    train(generator, discriminator, EPOCHS, criterion, optim_gen, optim_disc, loader)
    with torch.no_grad():
        chek = {
            'genearator': generator,
            'discriminator' : discriminator,
            'gen_state' : generator.state_dict(),
            'disc_state' : discriminator.state_dict(),
            'optim_gen' : optim_gen.state_dict(),
            'optim_disc' : optim_disc.state_dict(),
        }
        torch.save(chek, 'DGAN.pth.tar')
        print('saved successfully')