# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:56:59 2023

@author: sowmya
"""

from detection.object_detection import objectDetection
from segmentation.image_segmentation import imageSegmentation
from classification.image_classification import imageClassification
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

class YOLOv8:
    """
    Process images based on the specified task and subtask using YOLOv8.
    
    Args:
        task (str): Task to perform ('detect', 'segment', 'classify').
        subtask (str): Subtask ('crop', 'detect', 'segment').
        model_path (str): Path to the YOLOv8 model file.
        source (list): List of image paths.
        custom_class_colors (dict, optional): Dictionary of class-color pairs.
        priority_classes (list, optional): List of priority classes.
        plot (bool, optional): Whether to plot the results.
        white_background (bool, optional): Preference for segmented images.
        freq_table(bool, optional): Whether to plot the frequency table on the image
        freq_text_color(tuple, optional): text color for frequency table
        freq_table_color(tuple, optional): table color for frequency table
    
    Returns:
        list: Processed results based on the task and subtask.
        
    Task and Subtask Details:
        - Task 'detect' and Subtask 'detect':
            Returns a list of images with bounding boxes 
            for each image in the source.
    
        - Task 'detect' and Subtask 'crop':
            Returns a list of dictionaries with class:cropped_images 
            for each image in the source.
    
        - Task 'segment' and Subtask 'segment':
            Returns a list of images with masks and bounding boxes 
            for each image in the source.
    
        - Task 'segment' and Subtask 'crop':
            Returns a list of dictionaries with class:cropped_segmented_images 
            for each image in the source.
    
        - Task 'classify':
            Returns a list of 2 lists: [classes, confidences] 
            for each image in source.
    """
       
    def __init__(self, task, subtask, model_path, source, 
                 custom_class_colors=None, priority_classes=None, 
                 plot=False, white_background=False, freq_table=False, 
                 freq_text_color=(0,0,0), freq_table_color=(0,0,0)):
        self.task = task
        self.subtask = subtask
        self.model = YOLO(model_path)
        self.source = source
        self.priority_classes = priority_classes
        self.custom_class_colors = custom_class_colors
        self.plot = plot
        self.white_background = white_background
        self.freq_table = freq_table
        self.freq_table_color = freq_table_color
        self.freq_text_color = freq_text_color
        
        # Check if the source has supported image extensions
        if isinstance(source[0], str):
            self.check_supported_extensions(source)
            
        self.results = self.get_result()
        if self.plot: 
            plt.imshow(self.results[-1]);
            
        
    def check_supported_extensions(self, image_paths):
        """
        Check if the provided image paths have supported extensions.
    
        Args:
            image_paths (list): List of image paths.
    
        Returns:
            bool: True if all image paths have supported extensions, False otherwise.
        """
        # Supported image extensions for the YOLOv8 model
        SUPPORTED_EXTENSIONS = ['.bmp', '.dng', '.jpg', '.jpeg', '.png', '.mpo', 
                                '.tif', '.tiff', '.webp', '.pfm']
        
        for path in image_paths:
            _, extension = os.path.splitext(path)
            if extension.lower() not in SUPPORTED_EXTENSIONS:
                print(f"Unsupported image format for file: {path}")
                raise Exception("Please provide images with supported extensions.")


    def get_result(self):
        if self.task == 'detect':
            temp_obj = objectDetection(model=self.model, subtask=self.subtask,
                                      source=self.source,
                                      priority_classes=self.priority_classes,
                                      custom_class_colors=self.custom_class_colors,
                                      plot=self.plot, freq_table=self.freq_table, 
                                      freq_table_color=self.freq_table_color,
                                      freq_text_color=self.freq_text_color)
            result = temp_obj.process_result()

        elif self.task == 'segment':
            temp_obj = imageSegmentation(model=self.model, subtask=self.subtask,
                                      source=self.source,
                                      priority_classes=self.priority_classes,
                                      custom_class_colors=self.custom_class_colors,
                                      plot=self.plot,
                                      white_background=self.white_background,
                                      freq_table=self.freq_table, 
                                      freq_table_color=self.freq_table_color,
                                      freq_text_color=self.freq_text_color)
            result = temp_obj.process_result()

        elif self.task == 'classify':
            temp_obj = imageClassification(model=self.model, source=self.source, 
                                           plot=self.plot)
            class_names, confs = temp_obj.get_classname_conf()
            result = [class_names, confs]
        return result
    
   
if __name__ == "__main__":
    # Example Usage
    task = 'segment'
    subtask = 'segment'
    model_path = "path/to/model.pt"
    source = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
    custom_class_colors = {'person': (255, 0, 0), 'motorcycle': (0, 0, 255),
                           'dog': (0, 255, 0)}
    priority_classes = ['person', 'dog', 'motorcycle']
    plot = False
    white_background = False
    freq_table = False
    freq_text_color = (0,0,255)
    freq_table_color = (255,0,0)
    obj = YOLOv8(task, subtask, model_path, source,
                              custom_class_colors, priority_classes,
                              plot, white_background, freq_table, 
                              freq_text_color, freq_table_color)
    final_results = obj.results