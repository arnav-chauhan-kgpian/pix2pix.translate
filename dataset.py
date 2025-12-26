import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class AnimeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Filter to ensure we only load images (ignores .DS_Store or other system files)
        self.list_files = [
            f for f in os.listdir(self.root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        
        # Open image and force convert to RGB to handle PNG transparency issues
        image = np.array(Image.open(img_path).convert("RGB"))

        # Dynamic width splitting logic
        # Assumes the image is concatenated side-by-side (Input | Target)
        h, w, c = image.shape
        split_point = w // 2

        input_image = image[:, :split_point, :]
        target_image = image[:, split_point:, :]

        # Apply augmentations from config
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

if __name__ == "__main__":
    # Ensure the directory exists before running the test
    dataset = AnimeDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    for x, y in loader:
        print(f"Input Shape: {x.shape}")
        print(f"Target Shape: {y.shape}")
        
        save_image(x, "x_anime.png")
        save_image(y, "y_anime.png")
        
        import sys
        sys.exit()