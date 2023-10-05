# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 07:59:50 2023

@author: sowmya
"""

from utils.video_reader import read_video
from utils.freq_table import freqTable
from collections import Counter

from ultralytics import YOLO

import numpy as np
import cv2
import os

class videoProcessor:
    def __init__(self, video_path, vid_pathtype, model_path, output_video_path, 
                 custom_annotation=False, freq_table=False, hist_video=False,
                 priority_classes=None, custom_class_colors=None, task='detect',
                 freq_text_color=(0,0,0), freq_table_color=(0,0,0)):
        self.video_path = video_path
        self.vid_pathtype = vid_pathtype
        self.task = task
        self.model_path = model_path
        self.output_video_path = output_video_path
        self.custom_annotation = custom_annotation
        self.freq_table = freq_table
        self.hist_video = hist_video
        self.priority_classes = priority_classes
        self.custom_class_colors = custom_class_colors
        self.freq_text_color = freq_text_color
        self.freq_table_color = freq_table_color
        
        self.model = self.load_model()
        self.cap = read_video(video_path, vid_pathtype)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Get the original video's fps
        self.all_results = self.get_all_results(self.cap, self.model)
        self.classes, self.cls_frqs = self.get_all_frames_classes(self.all_results)

    
    def load_model(self):
        model = YOLO(self.model_path)
        return model
    

    def get_all_results(self, cap, model):
        # Loop through the video frames
        all_results = []
        while True:
            # Read a frame from the video
            ret, frame = self.cap.read()
            if ret:
                # Run YOLOv8 inference on the frame
                results = self.model(frame)
                all_results.append(results)
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        self.cap.release()
        return all_results
    
    
    def get_cls_frequency(self, results):
        class_names = results[0].names
        class_indices = results[0].boxes.cls.numpy().astype(int)
        classes = [class_names[idx] for idx in class_indices]
        occurrences = Counter(classes)
        return occurrences
    
    
    def get_all_frames_classes(self, all_results):
        cls_frqs = []
        for result in all_results:
            cls_frqs.append(self.get_cls_frequency(result))
        classes = list(set(cls for d in cls_frqs for cls in d))
        return classes, cls_frqs
    
    
    def plot_freq_table(self, freq_dict, image, classes):
        obj = freqTable(freq_dict=freq_dict, image_arr=image, 
                    video_frame=True, all_frames_classes=classes, 
                    text_color=self.freq_text_color, 
                    box_color=self.freq_table_color)
        return obj.plot_table()

    
    def get_annotated_frames(self):
        # Loop through the video frames
        annotated_frames = []
        for i in range(len(self.all_results)):
            # Visualize the results on the frame
            results = self.all_results[i]
            annotated_frame = results[0].plot()
            if self.freq_table: 
                annotated_frame = self.plot_freq_table(self.cls_frqs[i], 
                                            annotated_frame, self.classes)
            annotated_frames.append(annotated_frame)
        return annotated_frames
    
    
    def assign_cls_colors(self, classes):
        # Assign a random color to each class
        custom_class_colors = {}
        if self.custom_class_colors: 
            custom_class_colors=self.custom_class_colors
        
        for c in classes:
            if c not in custom_class_colors:
                custom_class_colors[c] = tuple(np.random.randint(0, 255, 3).tolist())
        return custom_class_colors
    
    
    def get_custom_annotated_frames(self):
        classes = self.classes
        if self.priority_classes: classes = self.priority_classes
        custom_class_colors = self.assign_cls_colors(classes)
        class_frqs = self.cls_frqs.copy()
        for dictionary in class_frqs:
            for class_name in classes:
                if class_name not in dictionary:
                    dictionary[class_name] = 0
                    
        # Loop through the video frames
        annotated_frames = []
        i=0
        while True:
            # Read a frame from the video
            ret, frame = self.cap.read()
            if ret:
                # Run YOLOv8 inference on the frame
                # import YOLOV8 from YOLOv8_toolkit's mypackage

                yolo_model = YOLOv8(self.task, self.task, self.model_path,
                                     source=[frame], 
                                     custom_class_colors=custom_class_colors,
                                     priority_classes=self.priority_classes)
                annotated_frame = yolo_model.get_result()
                
                if self.freq_table: 
                    annotated_frame = self.plot_freq_table(class_frqs[i], 
                                                annotated_frame, classes)
                annotated_frames.append(annotated_frame)
                i = i+1  
                
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        self.cap.release()
        return annotated_frames

    
    def get_final_video(self):
        if self.custom_annotation:
            annotated_frames = self.get_custom_annotated_frames()
        else:
            annotated_frames = self.get_annotated_frames()

        if self.hist_video: hist_frames = freq_hist_plot(annotated_frames)
        else: hist_frames = None

        return annotated_frames, fps, hist_frames
    
    def write_annotated_video(self, frames, output_video_path, fps):
        # Get the frame height and width from the first annotated frame
        frame_height, frame_width, _ = frames[0].shape

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose other codecs as needed
        out = cv2.VideoWriter(output_video_path, fourcc, fps,
                                (frame_width, frame_height))

        # Write each annotated frame to the video
        for frame in frames:
            out.write(frame)

        # Release the VideoWriter
        out.release()
        
        
    def export_video_outputs(self):
        annotated_frames, fps, hist_frames = self.get_final_video()
    
        output_video_path = os.path.join(self.output_video_path, 'final_output.mp4')
        self.write_annotated_video(annotated_frames, output_video_path, self.fps)
        
        if hist_frames:
            output_video_path = os.path.join(self.output_video_path, 'output_hist.mp4')
            self.write_annotated_video(hist_frames, output_video_path, self.fps)




# def freq_hist_plot(occurrences=None):
#     if occurrences:
#         plt.figure()
#         plt.hist(occurrences.keys(), occurrences.values())

