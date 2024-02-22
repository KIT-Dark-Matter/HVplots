import numpy as np
import cv2

def increase_brightness(img, value, color_mode='rgb'):
    if color_mode.lower() == 'rgb':
        to_hsv = cv2.COLOR_RGB2HSV
        from_hsv = cv2.COLOR_HSV2RGB
    elif color_mode.lower() == 'bgr':
        to_hsv = cv2.COLOR_BGR2HSV
        from_hsv = cv2.COLOR_HSV2BGR
    elif color_mode.lower() in ['gray', 'grayscale', 'grey', 'greyscale']:
        return np.clip(np.array(img, dtype=float)+value, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f'Got unknown color mode {color_mode}. Please choose from rgb, bgr or grayscale.')
    hsv = cv2.cvtColor(img, to_hsv)
    hsv[:,:,2] =  cv2.add(hsv[:,:,2], value)
    return cv2.cvtColor(hsv, from_hsv)

def increase_contrast(img, clip_limit, tile_grid_size=(8, 8), color_mode='rgb'):
    if color_mode.lower() == 'rgb':
        to_lab = cv2.COLOR_RGB2LAB
        from_lab = cv2.COLOR_LAB2RGB
    elif color_mode.lower() == 'bgr':
        to_lab = cv2.COLOR_BGR2LAB
        from_lab = cv2.COLOR_LAB2BGR
    else:
        raise ValueError(f'Got unknown color mode {color_mode}. Please choose from rgb, bgr')
    lab = cv2.cvtColor(img, to_lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, from_lab)

def standard_scale(img):
    return (img - np.mean(img)) / np.std(img)