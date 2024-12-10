from PIL import Image, ImageChops, ImageOps


def crop_to_content(img, image_size):
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
    target_height = max(cropped_height, int(cropped_width * (image_size[1] / image_size[0])))
    target_width = max(cropped_width, int(cropped_height * (image_size[0] / image_size[1])))

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


def add_background_to_image(img, bg_color):
    """Adds the frame's background color to a transparent image."""
    bg_image = Image.new("RGB", img.size, bg_color)
    bg_image.paste(img, (0, 0), img)
    return bg_image
