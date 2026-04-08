import torchvision.datasets
import torch
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

def download_dataset():
    """
    trainval_data_2007 = torchvision.datasets.VOCDetection(
        root='.', year='2007', image_set='trainval', download=False
    )

    trainval_data_2012 = torchvision.datasets.VOCDetection(root='.', year='2012', image_set='trainval', download=False)
    """
    val_dataset_2007 = torchvision.datasets.VOCDetection(root='.', year='2007', image_set='test', download=True)

class YOLODataset:
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        self.data_root_path = config.data_root_path
        self.years = config.years
        self.classes = config.classes
        
        if self.is_train:
            self.transform = A.Compose([
                A.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, scale=(0.8, 1.2), rotate=0, p=0.5),
                A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=30, p=0.5),
                A.Resize(config.image_size, config.image_size),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(config.image_size, config.image_size),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
        self.data = self.get_image_list(self.data_root_path, self.years)
        
    def get_image_list(self, root_path, years):
        image_list = []
        
        file_name = 'trainval.txt' if self.is_train else 'test.txt'
        
        for year in years:
            if not self.is_train and year == '2012': 
                continue
            
            txt_path = os.path.join(root_path, f'VOC{year}', 'ImageSets', 'Main', file_name)
            
            if not os.path.exists(txt_path):
                continue
            
            with open(txt_path, 'r') as f:
                ids = [line.strip() for line in f.readlines()]
                
            for img_id in ids:
                img_path = os.path.join(root_path, f'VOC{year}', 'JPEGImages', f'{img_id}.jpg')
                xml_path = os.path.join(root_path, f'VOC{year}', 'Annotations', f'{img_id}.xml')
                
                image_list.append((img_path, xml_path))
                
        return image_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        img_path, xml_path = self.data[i]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        # bboxes = [[x1, y1, width, height], ...]
        # class_labels = [0, 7, 4, ...]
        bboxes = []
        class_labels = []
        
        tree = ET.parse(xml_path)
        for obj in tree.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.classes: continue
            
            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)
            
            bboxes.append([x1, y1, x2 - x1, y2 - y1])
            class_labels.append(self.classes.index(class_name))
            
        
        transfromed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        image = transfromed['image']
        transformed_bboxes = transfromed['bboxes']
        transfromed_labels = transfromed['class_labels']
        
        S, B, C = self.config.S, self.config.B, self.config.C
        label_matrix = torch.zeros((S, S, B * 5 + C))
        
        for bbox, class_idx in zip(transformed_bboxes, transfromed_labels):
            class_idx = int(class_idx)
            
            x_center = (bbox[0] + bbox[2] / 2) / self.config.image_size
            y_center = (bbox[1] + bbox[3] / 2) / self.config.image_size
            width = bbox[2] / self.config.image_size
            height = bbox[3] / self.config.image_size

            grid_i = int(S * y_center)
            grid_j = int(S * x_center)
            
            if grid_i < S and grid_j < S and label_matrix[grid_i, grid_j, 4] == 0:
                label_matrix[grid_i, grid_j, 4] = 1
                
                x_relative = S * x_center - grid_j
                y_relative = S * y_center - grid_i
                
                label_matrix[grid_i, grid_j, 0:4] = torch.tensor([
                    x_relative, y_relative, width, height
                ])
                label_matrix[grid_i, grid_j, B * 5 + class_idx] = 1
                
        return image, label_matrix
    
    
if __name__ == "__main__":
    download_dataset()
    