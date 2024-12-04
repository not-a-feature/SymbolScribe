import os
from PIL import Image, ImageChops, ImageOps

# Configuration
base_dir = os.path.join(
    "C:",
    "\\",
    "Users",
    "Jules",
    "Documents",
    "SymbolScribe",
    "application",
    "symbols",
)

crop_size = 155


def process_image(fn, dataset_dir):
    print(fn)
    img_path = os.path.join(dataset_dir, "synthetic_dataset", fn)

    img = Image.open(img_path).convert("L")
    bg = Image.new(img.mode, img.size, 255)
    diff = ImageChops.difference(img, bg)

    # Extract the alpha channel based on the difference
    alpha = diff.convert("L")

    # Apply the new alpha channel to the image
    img.putalpha(alpha)

    width, height = img.size
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    img = img.crop((left, top, right, bottom))

    img.save(
        os.path.join(base_dir, img_path.replace("synthetic_dataset\\", ""))
    )  # Save directly into base_dir

    l_image = img.convert("L")
    # We must also crop the alpha channel
    alpha = alpha.crop((left, top, right, bottom))

    inverted_image = ImageOps.invert(l_image)
    inverted_image = Image.merge("LA", (inverted_image, alpha))

    name, ext = os.path.splitext(fn)
    inverted_image.save(os.path.join(base_dir, f"{name}_dark{ext}"))


for fn in os.listdir(os.path.join(base_dir, "synthetic_dataset")):
    if fn.endswith(".png"):
        process_image(fn, base_dir)
