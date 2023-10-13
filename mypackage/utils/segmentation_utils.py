# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:06:32 2023

@author: sowmya
"""

from utils.detection_utils import detectionUtils
from utils.color_utils import colorUtils
from utils.freq_table import freqTable
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import cv2

class segmentationUtils:
    def __init__(self, pred, priority_classes=None, freq_table=False,
                 custom_class_colors=None, plot=False, white_background=False,
                 freq_table_color=(0,0,0), freq_text_color=(0,0,0)):
        self.pred = pred
        self.xyxy_box_coords = pred.boxes.xyxy.numpy().astype(int)
        self.masks = pred.masks.xy
        self.confs = pred.boxes.conf.numpy()
        keys_list = pred.boxes.cls.numpy().astype(int).tolist()
        self.classes = [pred.names[key] for key in keys_list]
        self.img_shape = pred.orig_shape
        self.orig_rgb_img = pred.orig_img[:, :, ::-1].copy()
        self.custom_class_colors = custom_class_colors
        self.priority_classes = priority_classes
        self.plot = plot
        self.white_background = white_background
        self.freq_table = freq_table
        self.freq_table_color = freq_table_color
        self.freq_text_color = freq_text_color
        
    def segment_img(self, mask_points):
        height, width = self.img_shape[0], self.img_shape[1]
        mask_points = mask_points.astype(int)
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [mask_points], 255)
        masked_img = cv2.bitwise_and(self.orig_rgb_img, self.orig_rgb_img,
                                     mask=binary_mask)
        x, y, w, h = cv2.boundingRect(mask_points)
        segment_image = np.zeros((h, w, 3), dtype=np.uint8)
        segment_image[0:h, 0:w] = masked_img[y:y+h, x:x+w]
        if self.white_background:
            black_mask = np.all(segment_image == [0, 0, 0], axis=-1)
            segment_image[black_mask] = [255, 255, 255]
        return segment_image
    
    def crop_img(self):
        cropped_images = {}
        for index in range(len(self.masks)):
            class_label = self.classes[index]
            if self.priority_classes and class_label not in self.priority_classes:
                continue
            if class_label not in cropped_images: cropped_images[class_label]=[]

            mask = self.masks[index]
            mask_points = mask.astype(int)
            x, y, w, h = cv2.boundingRect(mask_points)
            rgb_crop = self.orig_rgb_img[y:y+h, x:x+w]
            if self.plot: plt.imshow(rgb_crop);
            cropped_images[class_label].append(rgb_crop)
        return cropped_images

    def crop_segmented_img(self):
        cropped_images = {}
        for index in range(len(self.masks)):
            class_label = self.classes[index]
            if self.priority_classes and class_label not in self.priority_classes:
                continue
            if class_label not in cropped_images: cropped_images[class_label]=[]

            mask = self.masks[index]
            rgb_crop = self.segment_img(mask)
            if self.plot: plt.imshow(rgb_crop);
            cropped_images[class_label].append(rgb_crop)
        return cropped_images

    def segmented_img_bbox(self, rgb_img, mask_points, color):
        height, width = self.img_shape[0], self.img_shape[1]
        mask_points = mask_points.astype(int)
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [mask_points], 255)
        masked_img = cv2.bitwise_and(rgb_img, rgb_img, mask=binary_mask)
        x, y, w, h = cv2.boundingRect(mask_points)
        x1, y1, x2, y2 = (x, y, x+w, y+h)
        box = [x1, y1, x2, y2]
        masked_img[binary_mask == 255] = color
        transparency = 0.8
        color_masked_img = cv2.addWeighted(rgb_img, 1.0,
                                   masked_img, transparency, 0)
        return box, color_masked_img


    def display_freq_table(self, img_arr):
        classes = self.classes
        if self.priority_classes:
            classes = [c for c in self.classes if c in self.priority_classes]
            
        obj = freqTable(freq_dict=Counter(classes), image_arr=img_arr, 
                        video_frame=False, all_frames_classes=None,
                        text_color=self.freq_text_color, 
                        box_color=self.freq_text_color)
        
        f_img = obj.plot_table();
        return f_img
    

    def obj_segment(self):
        rgb_img = self.orig_rgb_img
        # Dictionary to store colors for each class
        class_colors = self.custom_class_colors if self.custom_class_colors else {}
        for index in range(len(self.masks)):
            class_label = self.classes[index]
            if self.priority_classes and class_label not in self.priority_classes:
                continue
            conf = self.confs[index]
            text = f'{class_label}-{conf:.2f}'
            # Generate a random color for unseen classes
            if class_label not in class_colors:
                color = colorUtils.random_color()
                while True:
                    if color not in class_colors.values(): break
                    color = colorUtils.random_color()
                class_colors[class_label] = colorUtils.random_color()
            color = class_colors[class_label]
            mask = self.masks[index]
            box, masked_img = self.segmented_img_bbox(rgb_img, mask, color)
            rgb_img = detectionUtils.draw_annotated_boxes(masked_img, box,
                                                          color, text)
            
        if self.freq_table: rgb_img = self.display_freq_table(rgb_img)

        if self.plot: plt.imshow(rgb_img);
        return rgb_img
