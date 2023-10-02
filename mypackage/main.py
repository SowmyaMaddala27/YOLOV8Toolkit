# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:52:39 2023

@author: sowmya
"""

from yolo_results import YOLOv8
import os


def process_images(task, subtask, model_path, source, custom_class_colors=None,
                   priority_classes=None, plot=False, white_background=False):
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
    yolo_model = YOLOv8(task, subtask, model_path, source, custom_class_colors,
                       priority_classes, plot, white_background)
    results = yolo_model.get_result()
    return results



def check_supported_extensions(image_paths):
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
            return False
    return True

if __name__ == "__main__":
    # Example Usage
    task = 'classify'
    subtask = 'crop'
    model_path = "path/to/model.pt"
    source = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
    
    # Check if the source has supported image extensions
    if not check_supported_extensions(source):
        print("Please provide images with supported extensions.")
        exit(1)
    
    custom_class_colors = {'person': (255, 0, 0), 'motorcycle': (0, 0, 255),
                           'dog': (0, 255, 0)}
    priority_classes = ['person', 'dog', 'motorcycle']
    plot = False
    white_background = False

    processed_results = process_images(task, subtask, model_path, source,
                              custom_class_colors, priority_classes,
                              plot, white_background)