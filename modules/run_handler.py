import json
from datetime import datetime
import os
import re
import pandas as pd
import cv2
import pickle
from image_handler import Picture


PATH_TO_RUNDATA = os.path.join('C:', 'Users', 'valen' , 'Bachelorarbeit',  'hv_image_processing', 'run_data')

class Run:
    def __init__(self, hv_data_file: str) -> None:
        self.campaign = None
        self.run_id = None
        self.image_data = None
        self.file = hv_data_file
        column_headers = ['time', 'set_voltage', 'voltage', 'ramp', 'current', 'current_cap', 'warn', 'error']
        self.hv_data = pd.read_csv(self.file, delimiter='\t;', names=column_headers, usecols=[*range(8)], engine='python')
        datestring = self.get_date()
        self.hv_data['time'] = datestring + ' ' + self.hv_data['time'].astype(str)
        self.hv_data['time'] = pd.to_datetime(self.hv_data['time'])
        self.start_time = min(self.hv_data['time'])
        self.end_time = max(self.hv_data['time'])
        self.image_files = []
        self.pictures = []

    def associate_with_campaign(self, campaign: object):
        self.campaign = campaign
        return
    
    def get_date(self):
        if self.campaign is not None:
            return datetime.strftime(self.campaign.date, '%Y-%m-%d')
        date = re.match('.*(\d{2})\.(\d{2})\.(\d{4})\.csv', self.file.split(os.path.sep)[-1])
        if len(date.groups()) == 3:
            return f'{date.group(3)}-{date.group(2)}-{date.group(1)}'
        raise ValueError(f'Could not extract date from filename {self.file}.\nGot {date} as matching regex.')

    def collect_images_from_dir(self, directory):
        files = [f.path for f in os.scandir(directory) if f.path.endswith('JPG')]
        for file in files:
            pic = Picture(file=file)
            time = pic.time
            if self.start_time < time < self.end_time:
                self.image_files.append(file)
                self.pictures.append(pic)

    def set_run_id(self, idstring):
        self.run_id = idstring
        return
    
    def load_image_data(self):
        if self.run_id is None:
            raise ValueError("No run ID set for this run.")
        dict_file = os.path.join(PATH_TO_RUNDATA, f'{self.run_id}.pkl')
        if not os.path.exists(dict_file):
            print(f"No image data dict to load for {self.run_id}. Try run.calc_imagedata().")
            return False
        with open(dict_file, 'rb') as f:
            self.image_data = pickle.load(f)
        return True
    
    def calc_image_data(self):
        self.image_data = {'id': [], 'path': [], 'time': [], 'exposure': [], 'n_clusters': [], 'masses': [], 'areas': [], 'perimeters': [], 'cluster_positions': []}
        for i, f in enumerate(self.image_files):
            self.image_data['id'].append(i)
            self.image_data['path'].append(f)
            pic = self.pictures[i]
            self.image_data['time'].append(pic.time)
            self.image_data['exposure'].append(pic.exposure)
            pic.get_contour_data()
            self.image_data['n_clusters'].append(len(pic.contours))
            self.image_data['masses'].append(pic.contour_masses)
            self.image_data['areas'].append(pic.contour_areas)
            self.image_data['perimeters'].append(pic.contour_perimeters)
            self.image_data['cluster_positions'].append(pic.cluster_positions)

        if self.run_id is not None:
            dict_file = os.path.join(PATH_TO_RUNDATA, f'{self.run_id}.pkl')
            with open(dict_file, 'wb') as f:
                pickle.dump(self.image_data, f)
        return

class Campaign:
    def __init__(self, config) -> None:
        with open(config) as f:
            self.config = json.load(f)
        self.id = self.config['id']
        self.date = datetime.strptime(self.config['date'], '%Y-%m-%d')
        self.path_to_images = self.config['image_path']
        self.image_subdirs = self.config['image_subdirs']
        self.path_to_hvdata = self.config['hv_path']
        self.bright_imagefiles = [os.path.join(self.path_to_images, f) for f in self.config['bright_images']]

        self.runs = [Run(hv_data_file=f.path) for f in os.scandir(self.path_to_hvdata)]
        for i, run in enumerate(self.runs):
            run.associate_with_campaign(self)
            if self.image_subdirs:
                run.collect_images_from_dir(os.path.join(self.path_to_images, self.image_subdirs[i]))
            else:
                run.collect_images_from_dir(self.path_to_images)
            run.set_run_id(f'{self.id}_run{str(i).zfill(2)}')
        self.nruns = len(self.runs)
        if len(self.bright_imagefiles) == 1:
            self.bright_images = [cv2.imread(self.bright_imagefiles[0], cv2.IMREAD_COLOR)[:,:,::-1] for _ in self.runs]
        elif len(self.bright_imagefiles) == self.nruns:
            self.bright_images = [cv2.imread(f, cv2.IMREAD_COLOR)[:,:,::-1] for f in self.bright_imagefiles]