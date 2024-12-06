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


# --- Dataset Creation ---

# Define transformation functions
individual_transforms = [
    add_patches,
    distort,
    wave_distortion,
    add_noise,
    invert_colors,
]

# Generate all combinations of transform functions
all_combinations = []
for r in range(1, len(individual_transforms) + 1):
    for subset in itertools.combinations(individual_transforms, r):
        all_combinations.append(subset)

transform_names = ["original"] + [
    transform_func.__name__ for transform_func in individual_transforms
]
all_combinations = [tuple([lambda x: x])] + all_combinations

# Define combined transformations
transformations = []
for combination in all_combinations:
    if len(combination) == 1:
        transformations.append(combination[0])
    else:

        def combined_transform(img, combo=combination):
            transformed_img = img.copy()
            for transform in combo:
                transformed_img = transform(transformed_img)
            return transformed_img

        transformations.append(combined_transform)


# Function to apply transformations and save transformed images
def process_image(line, dataset_dir, num_samples):
    base_filename, symbol = line
    img_path = os.path.join(dataset_dir, base_filename)

    img = Image.open(img_path).convert("L")
    bg = Image.new(img.mode, img.size, 255)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox is None:
        bbox = (0, 0, 32, 32)
    img = img.crop(bbox)

    results = []
    for i in range(num_samples):
        for j, transform_func in enumerate(transformations):
            transformed_img = transform_func(img.copy())
            transform_flags = (
                [
                    1 if any(t.__name__ == transform_name for t in all_combinations[j]) else 0
                    for transform_name in transform_names
                ]
                if isinstance(all_combinations[j], tuple)
                else [
                    1 if all_combinations[j].__name__ == transform_name else 0
                    for transform_name in transform_names
                ]
            )
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
