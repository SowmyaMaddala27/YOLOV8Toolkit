# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:03:44 2023

@author: sowmya
"""

from utils.freq_table import freqTable
from utils.color_utils import colorUtils
from collections import Counter
import matplotlib.pyplot as plt
import cv2

class detectionUtils:
    def __init__(self, pred, priority_classes=None, freq_table=False,
                 custom_class_colors=None, plot=False, freq_table_color=(0,0,0),
                 freq_text_color=(0,0,0)):
        self.pred = pred
        self.xyxy_box_coords = pred.boxes.xyxy.numpy().astype(int)
        self.confs = pred.boxes.conf.numpy()
        keys_list = pred.boxes.cls.numpy().astype(int).tolist()
        self.classes = [pred.names[key] for key in keys_list]
        self.img_shape = pred.orig_shape
        self.orig_rgb_img = pred.orig_img[:, :, ::-1].copy()
        self.custom_class_colors = custom_class_colors
        self.priority_classes = priority_classes
        self.freq_table = freq_table
        self.freq_table_color = freq_table_color
        self.freq_text_color = freq_text_color
        self.plot = plot

    @staticmethod
    def draw_annotated_boxes(rgb_img, box, color, text):
        cv2.rectangle(rgb_img, (box[0], box[1]), (box[2], box[3]),
                      color, 5)
        # Define the font and scale of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        font_thickness = 2
        text_position = (box[0], box[1] - 10)

        # Get the size of the text to calculate the background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
                                    text, font, font_scale, font_thickness)

        # Calculate the position for the background rectangle
        background_rect_start = (box[0], box[1] - text_height - 10)
        background_rect_end = (box[0] + text_width, box[1] - baseline + 10)

        # Draw the background rectangle
        cv2.rectangle(rgb_img, background_rect_start,
                      background_rect_end, color, -1)

        # Put the text on the image
        rgb_img = cv2.putText(rgb_img, text, text_position,
                              font, font_scale, font_color, font_thickness)
        return rgb_img
    
    def crop_img(self):
        cropped_images = {}
        for index in range(len(self.xyxy_box_coords)):
            class_label = self.classes[index]
            if self.priority_classes and class_label not in self.priority_classes:
                continue
            if class_label not in cropped_images: cropped_images[class_label]=[]
            box = self.xyxy_box_coords[index]
            x1, y1, x2, y2 = (box[0], box[1], box[2], box[3])
            rgb_crop = self.orig_rgb_img[y1:y2, x1:x2]
            if self.plot: plt.imshow(rgb_crop);
            cropped_images[class_label].append(rgb_crop)
        return cropped_images


    def display_freq_table(self, img_arr):
        classes = self.classes
        if self.priority_classes:
            classes = [c for c in self.classes if c in self.priority_classes]
            
        obj = freqTable(freq_dict=Counter(classes), image_arr=img_arr, 
                        video=False, all_frames_classes=None,
                        text_color=self.freq_text_color, 
                        box_color=self.freq_text_color)
        
        f_img = obj.plot_table();
        return f_img
    

    def obj_detect(self):
        rgb_img = self.orig_rgb_img
        
        # Dictionary to store colors for each class
        class_colors = self.custom_class_colors if self.custom_class_colors else {}
        
        for index in range(len(self.xyxy_box_coords)):
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
            box = self.xyxy_box_coords[index]
            rgb_img = self.draw_annotated_boxes(rgb_img, box, color, text)
            
        if self.freq_table: rgb_img = self.display_freq_table(rgb_img)
        
        if self.plot: plt.imshow(rgb_img);
        
        return rgb_img