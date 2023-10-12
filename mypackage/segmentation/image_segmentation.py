# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:09:17 2023

@author: sowmya
"""

from utils.segmentation_utils import segmentationUtils

class imageSegmentation:
    def __init__(self, model, subtask, source, priority_classes=None,
                 custom_class_colors=None, plot=False, white_background=False,
                 freq_table=False, freq_table_color=(0,0,0), 
                 freq_text_color=(0,0,0)):
        self.model = model
        self.subtask = subtask
        self.source = source
        self.priority_classes = priority_classes
        self.custom_class_colors = custom_class_colors
        self.plot = plot
        self.white_background = white_background
        self.freq_table = freq_table
        self.freq_table_color = freq_table_color
        self.freq_text_color = freq_text_color
        self.predictions = self.model(self.source, conf=0.5, iou=0.7)

    def process_result(self):
        result = []
        for pred in self.predictions:
            classes = pred.boxes.cls
            if len(classes)==0: 
                result.append(pred.orig_img)
                continue
            
            obj = segmentationUtils(pred, self.priority_classes, self.freq_table,
                                self.custom_class_colors, self.plot,
                                self.white_background, self.freq_table_color, 
                                self.freq_text_color)
            if self.subtask=='crop':
                cropped_images = obj.crop_segmented_img()
                result.append(cropped_images)
            elif self.subtask=='segment':
                final_img = obj.obj_segment()
                result.append(final_img)
        return result
