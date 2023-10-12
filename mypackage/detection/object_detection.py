# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:58:11 2023

@author: sowmya
"""

from utils.detection_utils import detectionUtils

class objectDetection:
    def __init__(self, model, subtask, source, priority_classes=None,
                 custom_class_colors=None, plot=False, freq_table=False, 
                 freq_table_color=(0,0,0), freq_text_color=(0,0,0)):
        self.model = model
        self.subtask = subtask
        self.source = source
        self.priority_classes = priority_classes
        self.custom_class_colors = custom_class_colors
        self.plot = plot
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
            
            obj = detectionUtils(pred, self.priority_classes, self.freq_table,
                                self.custom_class_colors, self.plot,
                                self.freq_table_color, self.freq_text_color)
            if self.subtask=='crop':
                cropped_images = obj.crop_img()
                result.append(cropped_images)
            elif self.subtask=='detect':
                final_img = obj.obj_detect()
                result.append(final_img)
        return result