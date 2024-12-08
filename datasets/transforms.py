from torchvision import transforms        
import numpy as np

def get_transform(color_mode, input_shape): #, mean, std):
    if color_mode == "RGB":
        return transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_shape),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(
            #     brightness=0.2,
            #     contrast=0.2
            # ),
            # transforms.RandomInvert(),
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            
        ])
