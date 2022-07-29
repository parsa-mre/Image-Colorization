import config

import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import color
import torchvision.transforms.functional as TF


class FlowerDataset(Dataset):
    def __init__(self, root_dir, train=True, augment=config.AUGMENT):
        self.root_dir = os.path.join(
            root_dir, 'train') if train else os.path.join(root_dir, 'val')
        self.root_dir = os.path.join(self.root_dir, 'colored')
        self.list_files = sorted(os.listdir(self.root_dir))
        if '.DS_Store' in self.list_files:
            self.list_files.remove('.DS_Store')
        self.augment = augment and train

    def __len__(self):
        return len(self.list_files)

    def transform(self, image):

        lab = color.rgb2lab(image)
        lab = torch.tensor(lab, dtype=torch.float16)

        lab = torch.permute(lab, [2, 0, 1])
        if self.augment:
            # Random horizontal flipping
            if random.random() > 0.5:
                lab = TF.hflip(lab)

            # Random vertical flipping
            if random.random() > 0.5:
                lab = TF.vflip(lab)

        x = lab[0, :, :] / 255
        y = lab[1:3, :, :] / 128

        return x.reshape(1, 256, 256), y

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.list_files[index]))
        return self.transform(image)
