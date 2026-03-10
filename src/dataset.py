import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class OCTDataset(Dataset):

    def __init__(self, image_dir, img_size=128):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),  # CPU SAFE
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        images = Image.open(self.image_paths[idx])

        images = self.transform(images)
        images = (images - images.mean()) / (images.std() + 1e-8)

        return images
