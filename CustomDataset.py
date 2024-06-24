import glob
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.mobilenet_v2 import preprocess_input
from os import path, listdir
from torch.utils.data import Dataset, DataLoader
import os

class CustomDataset(Dataset):
    def __init__(self, image_paths, tfidf_paths, labels):
        super(CustomDataset, self).__init__()
        self.image_paths = image_paths
        self.tfidf_paths = tfidf_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image(self, img_path):
        # load image from path and convert to array
        img = load_img(img_path, color_mode="grayscale")
        st_data = img_to_array(img)
        img = preprocess_input(st_data)

        return img

    def _load_tfidf(self, tfidf_path):
        data = np.load(tfidf_path)

        return data
    
    def __getitem__(self, idx):
        # file 경로
        img = self._load_image(self.image_paths[idx])
        tfidf = self._load_tfidf(self.tfidf_paths[idx])
        # label 생성
        lbl = self.labels[idx]
        # image, label return
        return  img.astype(np.float32), tfidf.astype(np.float32), lbl