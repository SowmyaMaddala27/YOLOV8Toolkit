# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:08:23 2023

@author: sowmya
"""

from utils.classification_utils import classificationUtils

class imageClassification:
    def __init__(self, model, source, plot=False):
        self.model = model
        self.source = source
        self.plot = plot

        self.predictions = self.model(self.source)

    def get_classname_conf(self):
        class_names = []
        confs = []
        for pred in self.predictions:
            obj = classificationUtils(pred, self.plot)
            class_name, conf = obj.get_classname_conf(pred)
            class_names.append(class_name)
            confs.append(conf)
        return class_names, confs

    def process_result(self):
        result = []
        for pred in self.predictions:
            obj = classificationUtils(pred, self.plot)
            class_name, conf = obj.get_classname_conf(pred)
            result.append([class_name, conf])
        return result
