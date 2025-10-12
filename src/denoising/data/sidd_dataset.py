import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 as cv
import os

class SIDD(Dataset):
    def __init__(self, root_dir, mode = 'train', img_size = 128):
        """
        Args:
            root_dir: Dir from 'noisy' and 'clean' dataset
            mode: 'train' or 'test'
            img_size: Size of images
        """

        self.noisy_dir = os.path.join(root_dir, 'noisy')
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.img_size = img_size
        self.mode = mode

        self.image_names = sorted(os.listdir(self.noisy_dir))
        self.transform = self._build_transforms()

    def _build_transforms(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),  #to [0,1]
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.image_names[idx])
        clean_path = os.path.join(self.clean_dir, self.image_names[idx])

        noisy_img = cv.imread(noisy_path, cv.IMREAD_COLOR_RGB)
        clean_img = cv.imread(clean_path, cv.IMREAD_COLOR_RGB)

        if noisy_img is None or clean_img is None:
            raise FileNotFoundError(f"Error loading {noisy_path} or {clean_path}")

        noisy_tensor = self.transform(noisy_img)
        clean_tensor = self.transform(clean_img)

        return noisy_tensor, clean_tensor