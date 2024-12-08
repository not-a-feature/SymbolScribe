import os
import csv
import random
import itertools
import numpy as np
from PIL import Image, ImageOps, ImageChops
from skimage.transform import AffineTransform, warp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Set random seed for reproducibility
random.seed(123452345234)

# Configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "mixed_dataset")
csv_file = os.path.join(base_dir, "mixed_dataset.csv")
manual_dir = os.path.join(base_dir, "manual_dataset")
manual_csv_file = os.path.join(base_dir, "manual_dataset.csv")
synthetic_dir = os.path.join(base_dir, "synthetic_dataset")
synthetic_csv_file = os.path.join(base_dir, "synthetic_dataset.csv")

number_samples_manual = 4
number_samples_synthetic = 1

crop_padding = 50
min_patches = 4
max_patches = 6
patch_size_range = (0.1, 0.2)
angle_range = (-15, 15)
amplitude_range = (4, 8)
period_range = (40, 60)


# --- Transformation Functions with Random Parameters ---
def distort(img, shear_range=(-0.4, 0.4)):
    shear = random.uniform(*shear_range)
    transform = AffineTransform(shear=shear)
    distorted_array = warp(np.array(img), transform, mode="constant", cval=1)

    img = Image.fromarray((distorted_array * 255).astype(np.uint8)).convert("L")
    angle = random.uniform(*angle_range)
    rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
    return rotated_img


def add_noise(img, amount=0.5):
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    num_pixels_to_change = int(height * width * amount)

    # Create indices for pixels to change
    indices = np.random.choice(height * width, num_pixels_to_change, replace=False)
    row_indices = indices // width
    col_indices = indices % width

    # Randomly assign black or white to the chosen pixels
    colors = np.random.choice([0, 255], num_pixels_to_change)
    img_array[row_indices, col_indices] = colors

    noisy_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array).convert("L")


def wave_distortion(img):
    amplitude = random.uniform(*amplitude_range)
    period = random.uniform(*period_range)
    img_array = np.array(img)
    width, height = img_array.shape
    for y in range(height):
        offset = int(amplitude * np.sin(2 * np.pi * y / period))
        img_array[:, y] = np.roll(img_array[:, y], offset, axis=0)
    return Image.fromarray(img_array).convert("L")


def add_patches(img):
    num_patches = random.randint(min_patches, max_patches)
    img_copy = img.copy()
    for _ in range(num_patches):
        patch_width = random.randint(
            int(patch_size_range[0] * img.width),
            int(patch_size_range[1] * img.width),
        )
        patch_height = random.randint(
            int(patch_size_range[0] * img.height),
            int(patch_size_range[1] * img.height),
        )
        patch = Image.new("L", (patch_width, patch_height), color="black")
        x = random.randint(0, img.width - patch_width)
        y = random.randint(0, img.height - patch_height)
        img_copy.paste(patch, (x, y))
    return img_copy


def invert_colors(img):
    return ImageOps.invert(img)


def original(img):
    return img


# Define transformation functions
individual_transforms = [
    add_patches,
    distort,
    wave_distortion,
    add_noise,
    invert_colors,
]

# Generate all combinations of transform functions
transformations = []
for r in range(1, len(individual_transforms) + 1):
    for subset in itertools.combinations(individual_transforms, r):
        transformations.append(subset)

transformations = [tuple([original])] + transformations

transform_names = [transform_func.__name__ for transform_func in individual_transforms]
transform_names = ["original"] + transform_names


# Function to apply transformations and save transformed images
def process_image(line, dataset_dir, num_samples):
    base_filename, symbol = line
    img_path = os.path.join(dataset_dir, base_filename)

    # Crop to symbol with padding
    img = Image.open(img_path).convert("L")
    bg = Image.new(img.mode, img.size, 255)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()

    if bbox is None:
        # Edge case for Empty image
        bbox = (0, 0, 32, 32)

    cropped_img = img.crop(bbox)

    patch_bbox = (
        max(0, bbox[0] - crop_padding),  # Left
        max(0, bbox[1] - crop_padding),  # Upper
        min(bbox[2] + crop_padding, img.width),  # Right
        min(bbox[3] + crop_padding, img.height),  # Lower
    )

    cropped_img_for_patches = img.crop(patch_bbox)

    results = []
    for i in range(num_samples):
        for j, transform_combo in enumerate(transformations):
            if transform_combo[0].__name__ == "add_patches":
                transformed_img = cropped_img_for_patches.copy()
            else:
                transformed_img = cropped_img.copy()

            for transform_func in transform_combo:
                transformed_img = transform_func(transformed_img)
                if transform_func.__name__ == "add_patches":
                    bg = Image.new(transformed_img.mode, transformed_img.size, 255)
                    diff = ImageChops.difference(transformed_img, bg)
                    bbox = diff.getbbox()
                    transformed_img = transformed_img.crop(bbox)

            transform_flags = [
                1 if any(t.__name__ == transform_name for t in transform_combo) else 0
                for transform_name in transform_names
            ]
            cleaned_name = base_filename.removesuffix(".png")
            transformed_filename = f"{cleaned_name}_{i}_{j}.png"
            transformed_img.save(os.path.join(output_dir, transformed_filename))
            results.append([transformed_filename, symbol] + transform_flags)
    return results


# Generate the dataset in parallel
def generate(lines, dataset_dir, num_samples):
    with ThreadPoolExecutor(max_workers=18) as executor:
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            futures = [
                executor.submit(process_image, line, dataset_dir, num_samples) for line in lines
            ]
            for future in tqdm(futures):
                for row in future.result():
                    writer.writerow(row)


# --- Dataset Creation ---
# Setup output CSV file
os.makedirs(output_dir, exist_ok=True)
with open(csv_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "latex_symbol"] + transform_names)

# Process manual and synthetic datasets
with open(manual_csv_file, "r") as f:
    lines = [line.split(",") for l in f.readlines() if (line := l.removesuffix("\n"))][1:]
generate(lines, manual_dir, number_samples_manual)

with open(synthetic_csv_file, "r") as f:
    lines = [line.split(",") for l in f.readlines() if (line := l.removesuffix("\n"))][1:]
generate(lines, synthetic_dir, number_samples_synthetic)

print(f"Dataset created in '{output_dir}'")
