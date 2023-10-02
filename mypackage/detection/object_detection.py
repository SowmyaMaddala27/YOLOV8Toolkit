# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:58:11 2023

@author: sowmya
"""

from utils.detection_utils import detectionUtils

class objectDetection:
    def __init__(self, model, subtask, source, priority_classes=None,
                 custom_class_colors=None, plot=False):
        self.model = model
        self.subtask = subtask
        self.source = source
        self.priority_classes = priority_classes
        self.custom_class_colors = custom_class_colors
        self.plot = plot

        self.predictions = self.model(self.source, conf=0.5, iou=0.7)

    def process_result(self):
        result = []
        for pred in self.predictions:
            obj = detectionUtils(pred, self.priority_classes,
                                self.custom_class_colors, self.plot)
            if self.subtask=='crop':
                cropped_images = obj.crop_img()
                result.append(cropped_images)
            elif self.subtask=='detect':
                final_img = obj.obj_detect()
                result.append(final_img)
        return result