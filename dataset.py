import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
from symbols import symbols
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image, ImageChops, ImageOps

label_mapping = {symbol[0]: i for i, symbol in enumerate(symbols)}

# Dataset configuration
image_size = (32, 32)  # Width / Height

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
)


def crop_to_content(img):
    bg = Image.new(img.mode, img.size, 255)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()

    if bbox is None:
        # Edge case for empty image
        return None

    # Crop to content
    cropped_img = img.crop(bbox)

    # Calculate dimensions to maintain width / height aspect ratio
    cropped_width, cropped_height = cropped_img.size
    target_height = max(cropped_height, cropped_width)
    target_width = max(cropped_width, cropped_height)

    # Add padding to center the cropped content
    pad_left = (target_width - cropped_width) // 2
    pad_top = (target_height - cropped_height) // 2
    pad_right = target_width - cropped_width - pad_left
    pad_bottom = target_height - cropped_height - pad_top

    padded_img = ImageOps.expand(
        cropped_img,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill="white",
    )
    return padded_img


class SymbolDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        crop_to_content=True,
        load_directly=False,
        num_workers=None,
    ):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.crop_to_content = crop_to_content
        self.load_directly = load_directly
        self.num_workers = num_workers if num_workers else os.cpu_count() - 1

        self.labels = [label_mapping[str(label)] for label in self.data.iloc[:, 1]]

        self.transform = transform

        print(f"Dataset with {len(self.data)} samples.")
        if load_directly:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                self.loaded_data = list(
                    tqdm(
                        executor.map(self.load, range(len(self.data))),
                        total=len(self.data),
                        desc="Loading Images",
                    )
                )
        else:
            print("Data will be loaded on demand.")

    def load(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        try:
            image = Image.open(img_name)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, 0, 0, None

        if self.crop_to_content:
            try:
                image = crop_to_content(image)
            except Exception as e:
                print(f"Error cropping image {img_name}: {e}")
                return None, 0, 0, None

        width, height = image.size
        try:
            image = self.transform(image)
        except Exception as e:
            print(f"Error transforming image {img_name}: {e}")
            return None, 0, 0, None

        label = self.labels[idx]
        return image, width, height, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.load_directly:
            return self.loaded_data[idx]
        else:
            return self.load(idx)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "mixed_dataset")
    csv_path = os.path.join(base_dir, "mixed_dataset.csv")

    dataset = SymbolDataset(
        csv_file=csv_path,
        root_dir=dataset_dir,
        crop_to_content=False,
        load_directly=True,
    )
    torch.save(dataset, os.path.join(base_dir, "dataset.pth"))
