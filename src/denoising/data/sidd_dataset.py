import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2 as cv
from PIL import Image
import os, random

class SIDD(Dataset):
    def __init__(self, root_dir, mode = 'train', img_size = 128, patches = 50):
        """
        Args:
            root_dir: Path containing 'noisy' and 'clean' subdirectories.
            mode: 'train' or 'test'
            img_size: Size of images
            patches: Number of random crops extracted from each image.
        """

        self.noisy_dir = os.path.join(root_dir, 'noisy')
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.img_size = img_size
        self.mode = mode
        self.patches_img = patches

        self.image_names = sorted(os.listdir(self.noisy_dir))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_names) * self.patches_img
    
    def _read_rgb(self, path):
        bgr = cv.imread(path, cv.IMREAD_COLOR)
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def __getitem__(self, idx):
        img_idx = idx // self.patches_img
        noisy_path = os.path.join(self.noisy_dir, self.image_names[img_idx])
        clean_path = os.path.join(self.clean_dir, self.image_names[img_idx])

        noisy = self._read_rgb(noisy_path)
        clean = self._read_rgb(clean_path) 

        if self.mode == 'train':
            th, tw = self.img_size, self.img_size
            w, h = noisy.size
            if h < th or w < tw:
                crop = transforms.CenterCrop((max(th, h), max(tw, w)))
                noisy = crop(noisy); clean = crop(clean)

            i, j, h, w = transforms.RandomCrop.get_params(noisy, output_size=(th, tw))
            noisy = TF.crop(noisy, i, j, h, w)
            clean = TF.crop(clean, i, j, h, w)

            if random.random() < 0.5:
                noisy = TF.hflip(noisy); clean = TF.hflip(clean)
            if random.random() < 0.5:
                noisy = TF.vflip(noisy); clean = TF.vflip(clean)
        else:
            crop = transforms.CenterCrop(self.img_size)
            noisy = crop(noisy)
            clean = crop(clean)

        noisy_tensor = self.to_tensor(noisy) #[0,1]
        clean_tensor = self.to_tensor(clean) #[0,1]

        return noisy_tensor, clean_tensor