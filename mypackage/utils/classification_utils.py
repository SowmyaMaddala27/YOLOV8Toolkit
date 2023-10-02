# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:05:02 2023

@author: sowmya
"""

import matplotlib.pyplot as plt
import cv2

class classificationUtils:
    def __init__(self, pred, plot=False):
        if plot:
            img = cv2.cvtColor(pred.orig_img, cv2.COLOR_BGR2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # White color
            font_thickness = 2
            class_name, conf = self.get_classname_conf(pred)
            text = f'{class_name}: {conf}'
            (text_width, text_height), _ = cv2.getTextSize(text, font,
                                                    font_scale, font_thickness)
            image_height, image_width, _ = img.shape
            x = image_width - text_width - 20
            y = 50
            # Create a background rectangle for the text
            background_color = (0, 0, 0)
            background_end = (x + text_width, y - text_height-20)
            cv2.rectangle(img, (x, y), background_end, background_color, -1)

            # Add the text to the image
            image_with_text = cv2.putText(img, text, (x, y), font, font_scale,
                                          font_color, font_thickness)
            plt.figure()
            plt.axis('off')
            plt.title(text)
            plt.imshow(image_with_text);

    def get_classname_conf(self, pred):
        class_name = pred.names[pred.probs.top1]
        conf = f'{pred.probs.top1conf.item():.2f}'
        return class_name, conf
