"""Image composition and modification primitives"""

from PIL import Image, ImageEnhance
from renderer.utils import *

def genImage(lib, sample):
    """Generate image, labels, and meta data from a sample"""

    # Handle both variable-length (1-2 cars) and fixed-length (2 cars) arrays
    # Also handle samples from error table where cars might be a list
    cars = sample.cars if hasattr(sample, 'cars') else []
    
    # Filter out None/invalid cars (can happen with fixed-length arrays or error table samples)
    valid_cars = []
    for car in cars:
        # Check if car has required attributes and they're not None
        if (hasattr(car, 'yPos') and hasattr(car, 'xPos') and hasattr(car, 'carID') and
            car.yPos is not None and car.xPos is not None and car.carID is not None):
            valid_cars.append(car)
    
    # Sort valid cars by depth
    sorted_cars = sorted(valid_cars, key=lambda car: -car.yPos[0])
    
    fg = []
    for car in sorted_cars:
        fg += [fgObj(fgId=car.carID,
                     x=car.xPos[0],
                     y=car.yPos[0])]
    
    # Handle parameters that might be None (from error table)
    def safe_get(attr_name, default=1.0):
        val = getattr(sample, attr_name, None)
        if val is None:
            return default
        # Handle both scalar and array-like values
        extracted = val[0] if hasattr(val, '__getitem__') else val
        # Check if the extracted value is None (e.g., (None,) tuple)
        if extracted is None:
            return default
        return extracted

    return genCompImg(
        lib,
        fg,
        bgId=sample.backgroundID if hasattr(sample, 'backgroundID') else 0,
        brightness=safe_get('brightness', 1.0),
        sharpness=safe_get('sharpness', 1.0),
        contrast=safe_get('contrast', 1.0),
        color=safe_get('color', 1.0))


def scaleImg(img, scale):
    return img.resize((np.array(img.size) * scale).astype(int))


def scaleGetLoc(img, scale, centroid):
    scaledImg = scaleImg(img, scale)
    topRightLoc = (
        np.array(centroid) - np.array(scaledImg.size) * 0.5).astype(int)
    return scaledImg, topRightLoc


def modifyImageBscc(imageData, brightness, sharpness, contrast, color):
    """Update with brightness, sharpness, contrast and color."""
    
    # Ensure all parameters are valid numbers, default to 1.0 if None
    brightness = 1.0 if brightness is None else float(brightness)
    sharpness = 1.0 if sharpness is None else float(sharpness)
    contrast = 1.0 if contrast is None else float(contrast)
    color = 1.0 if color is None else float(color)

    brightnessMod = ImageEnhance.Brightness(imageData)
    imageData = brightnessMod.enhance(brightness)

    sharpnessMod = ImageEnhance.Sharpness(imageData)
    imageData = sharpnessMod.enhance(sharpness)

    contrastMod = ImageEnhance.Contrast(imageData)
    imageData = contrastMod.enhance(contrast)

    colorMod = ImageEnhance.Color(imageData)
    imageData = colorMod.enhance(color)

    return imageData


def genCompImg(library,
               fgObjects,
               bgId=0,
               brightness=1.,
               sharpness=1.,
               contrast=1.,
               color=1.):
    """Compose an image from a sample."""

    background = library.backgroundObjects[bgId]
    scalingFactor = background.scaling
    backgroundCopy = background.image.copy()

    # remove alpha channel from background (if present)
    if backgroundCopy.mode in (
            'RGBA', 'LA') or (backgroundCopy.mode == 'P'
                              and 'transparency' in backgroundCopy.info):
        backgroundNoAlpha = Image.new("RGB", backgroundCopy.size,
                                      (255, 255, 255))
        backgroundNoAlpha.paste(
            backgroundCopy,
            mask=backgroundCopy.split()[3])  # 3 is the alpha channel
    else:
        backgroundNoAlpha = backgroundCopy

    # Add foreground images
    boxes = []
    for i, fgi in zip(range(len(fgObjects)), fgObjects):
        x, y, fg = fgi.x, fgi.y, fgi.fgId
        scaleFg = y * (
            scalingFactor.back - scalingFactor.front) + scalingFactor.front
        sampleConvSpace = ld2bbSample(sample=[x, y], h=background.homographyH)
        foreground = library.foregroundObjects[fg]
        scaledImg, topRightLoc = scaleGetLoc(foreground.image, scaleFg,
                                             sampleConvSpace)

        # paste car
        backgroundNoAlpha.paste(scaledImg, tuple(topRightLoc), scaledImg)

        # store labels
        intCentroid = list(sampleConvSpace.astype(int))
        listSize = list(scaledImg.size)

        boxes.append(intCentroid + listSize)

    modifImg = modifyImageBscc(
        imageData=backgroundNoAlpha,
        brightness=brightness,
        sharpness=sharpness,
        contrast=contrast,
        color=color)

    return modifImg, boxes