import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, transform=None, color_mode="RGB"):
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.transform = transform
        self.color_mode = color_mode
        self.img_labels = []

        for img_dir in self.img_dirs:
            self.img_labels.extend([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpeg', '.png', '.bmp', '.JPG'))])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels[idx]
        image = Image.open(img_path).convert(self.color_mode)
        if self.transform:
            image = self.transform(image)
        return image
