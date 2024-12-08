from PIL import Image, ImageChops


def crop_to_content(image):
    """Crops image to the non-white content."""
    bg = Image.new(image.mode, image.size, 255)
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox is None:
        # Edge case for Empty image
        bbox = (0, 0, 32, 32)

    image = image.crop(bbox)
    return image


def add_background_to_image(img, bg_color):
    """Adds the frame's background color to a transparent image."""
    bg_image = Image.new("RGB", img.size, bg_color)
    bg_image.paste(img, (0, 0), img)
    return bg_image
