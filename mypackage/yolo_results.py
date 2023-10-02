# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:56:59 2023

@author: sowmya
"""

from detection.object_detection import objectDetection
from segmentation.image_segmentation import imageSegmentation
from classification.image_classification import imageClassification
from ultralytics import YOLO

class YOLOv8:
    def __init__(self, task, subtask, model_path, source, 
                 custom_class_colors=None, priority_classes=None, 
                 plot=False, white_background=False):
        self.task = task
        self.subtask = subtask
        self.model = YOLO(model_path)
        self.source = source
        self.priority_classes = priority_classes
        self.custom_class_colors = custom_class_colors
        self.plot = plot
        self.white_background = white_background

    def get_result(self):
        if self.task == 'detect':
            temp_obj = objectDetection(model=self.model, subtask=self.subtask,
                                      source=self.source,
                                      priority_classes=self.priority_classes,
                                      custom_class_colors=self.custom_class_colors,
                                      plot=self.plot)
            result = temp_obj.process_result()

        elif self.task == 'segment':
            temp_obj = imageSegmentation(model=self.model, subtask=self.subtask,
                                      source=self.source,
                                      priority_classes=self.priority_classes,
                                      custom_class_colors=self.custom_class_colors,
                                      plot=self.plot,
                                      white_background=self.white_background)
            result = temp_obj.process_result()

        elif self.task == 'classify':
            temp_obj = imageClassification(model=self.model, source=self.source, 
                                           plot=self.plot)
            class_names, confs = temp_obj.get_classname_conf()
            result = [class_names, confs]
        return result