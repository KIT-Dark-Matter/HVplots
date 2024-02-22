import os
from datetime import timedelta
import cv2
import rawpy
import exifread
import pandas as pd
import numpy as np
import pickle
import skimage
from scipy.ndimage import gaussian_filter

import processing   #'from . ' at the start was removed

'''
Image time has an offset! See calibration image.
Real time = 13:29:20
Tag time  = 13:32:11
'''
IMAGE_DT = timedelta(minutes=2, seconds=51)

SHAPE = (5568, 3712, 3)

GLOBAL_CACHE_DATA_FLAG = True
# Manually create a cache directory. Checking and automatically creating it for every pic takes time.
PATH_TO_CACHE = os.path.join('C:', 'Users', 'valen' , 'Bachelorarbeit',  'hv_image_processing', 'cache')

class Picture:
    def __init__(self, file) -> None:
        self.file = file
        self.save_data = GLOBAL_CACHE_DATA_FLAG
        # self.load_image()
        self.run = None
        with open(self.file, 'rb') as f:
            self.tags = exifread.process_file(f)
        self.time = pd.to_datetime(str(self.tags['EXIF DateTimeOriginal']), format='%Y:%m:%d %H:%M:%S') - IMAGE_DT
        self.exposure = float(eval(str(self.tags['EXIF ExposureTime'])))
        self.contours = []
        self.contour_masses = []
        self.contour_areas = []
        self.contour_perimeters = []
        self.cluster_positions = []

    def load_image(self):
        ''' Dont save the img. Too much memory for ~500 imgs. '''
        if self.file[-3:].lower() == 'cr2':
            with rawpy.imread(self.file) as raw:
                return raw.postprocess()
        else:
            return cv2.imread(self.file, cv2.IMREAD_COLOR)[:,:,::-1]

    def associate_with_run(self, run):
        self.run = run

    def get_processed_image(self, gauss_sigma=3, contrast_clip=2, contrast_grid_size=(50, 50)):
        ''' Dont save the img. Too much memory for ~500 imgs. '''
        img = self.load_image()
        img = filter_circle(img) #filter_green_arrows(img)
        img = gaussian_filter(img, sigma=gauss_sigma)
        img = processing.increase_contrast(img.astype(np.uint8), clip_limit=contrast_clip, tile_grid_size=contrast_grid_size)
        img = cv2.cvtColor(processing.increase_brightness(img, value=0), cv2.COLOR_RGB2GRAY)
        return img
    
    def calc_contour_data(self, binary_threshold=50):
        img = self.get_processed_image()
        self.contours = get_cluster_contours(img=img, binary_threshold=binary_threshold)
        for contour_id, contour in enumerate(self.contours):
            mass, area, perimeter = get_contour_values(img, contour)
            self.contour_masses.append(mass)
            self.contour_areas.append(area)
            self.contour_perimeters.append(perimeter)
            self.cluster_positions.append((np.mean(contour[:,:,0]), np.mean(contour[:,:,1])))
        if self.save_data:
            savedir = PATH_TO_CACHE
            savename = self.file.split(os.sep)[-1].replace('.JPG', '_data.pkl')
            data = dict(contours   = self.contours,
                        masses     = self.contour_masses,
                        areas      = self.contour_areas,
                        perimeters = self.contour_perimeters,
                        positions  = self.cluster_positions)
            with open(os.path.join(savedir, savename), 'wb') as f:
                pickle.dump(data, f)
        return
    
    def load_contour_data(self):
        savedir = PATH_TO_CACHE
        savename = self.file.split(os.sep)[-1].replace('.JPG', '_data.pkl')
        with open(os.path.join(savedir, savename), 'rb') as f:
            data = pickle.load(f)
        self.contours = data['contours']
        self.contour_masses = data['masses']
        self.contour_areas = data['areas']
        self.contour_perimeters = data['perimeters']
        self.cluster_positions = data['positions']
        return

    def get_contour_data(self):
        savedir = PATH_TO_CACHE
        savename = self.file.split(os.sep)[-1].replace('.JPG', '_data.pkl')
        if os.path.isfile(os.path.join(savedir, savename)):
            self.load_contour_data()
        else:
            self.calc_contour_data()
        return


def find_green_arrows(img):
    img = img.astype(np.float32)
    mask = np.where(img[:,:,1] - 0.7*(img[:,:,0]+img[:,:,2]) > 10, 1, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours

def filter_arrow_contours(contours):
    resolution = 500
    x = np.array([500, 2000, 3400])
    y = np.array([4150, 4850, 1250])
    matching_contours = []
    for c in contours:
        av_x = np.average(c[:,:,0])
        av_y = np.average(c[:,:,1])
        for i in range(3):
            if x[i]-resolution < av_x < x[i]+resolution:
                if y[i]-resolution < av_y < y[i]+resolution:
                    matching_contours.append(c)
                    break
    return matching_contours

def filter_green_arrows(img):
    contours = find_green_arrows(img)
    contours = filter_arrow_contours(contours)
    masked = np.copy(img)
    for c in contours:
        cv2.drawContours(masked, [c], 0, color=0, thickness=-1)
    return masked

def filter_circle(img):
    contours = find_green_arrows(img)
    contours = filter_arrow_contours(contours)
    xs = [np.mean(c[:,:,0]) for c in contours]
    ys = [np.mean(c[:,:,1]) for c in contours]
    # sort by x
    pos = [(x,y) for x, y in sorted(zip(xs, ys))]
    # in row, col
    if len(pos) < 2:
        center = (img.shape[0]//2, img.shape[1]//2)
    else:
        center = ((pos[0][1] + pos[-1][1])/2, (pos[0][0] + pos[-1][0])/2)
    rr, cc = skimage.draw.disk(center=center, radius=img.shape[1]//2 + 10, shape=img.shape[:-1])
    mask = np.ones_like(img)
    mask[rr,cc] = False
    masked = np.where(mask, np.array([0,0,0]), img)
    return masked

def get_cluster_contours(img, binary_threshold=50):
    _, binary = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    binary = np.uint8(binary)
    contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours

def get_contour_pixels(img, contour):
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [contour], 0, color=1, thickness=-1)
    pixels = img[np.where(mask==1)]
    return pixels

def get_contour_values(img, contour):
    perimeter = cv2.arcLength(contour, True)
    pixels = get_contour_pixels(img, contour)
    area = len(pixels)
    mass = sum(pixels)
    return mass, area, perimeter