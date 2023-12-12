# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 00:01:24 2023

@author: sowmya
"""

# Import necessary libraries

from yolo_results import YOLOv8
import cv2

from tqdm import tqdm
import os


class copy_data:
    def __init__(self, source_path, des_dir, task, subtask, model_path, 
                 custom_class_colors=None, priority_classes=None, 
                 plot=False, white_background=False, freq_table=False, 
                 freq_text_color=(0,0,0), freq_table_color=(0,0,0)):
        self.source_path = source_path
        self.des_dir = des_dir
        self.task = task
        self.subtask = subtask
        self.model_path = model_path
        self.priority_classes = priority_classes
        self.custom_class_colors = custom_class_colors
        self.plot = plot
        self.white_background = white_background
        self.freq_table = freq_table
        self.freq_table_color = freq_table_color
        self.freq_text_color = freq_text_color
        # self.save_pipeline2()
        
    def get_filepaths(self):
        file_paths = []
        if os.path.isdir(self.source_path):
            for file in os.listdir(self.source_path):
                filepath = os.path.join(self.source_path, file)
                if os.path.isfile(filepath):
                    if cv2.imread(filepath) is not None:
                        file_paths.append(filepath)
        elif os.path.isfile(self.source_path):
            if cv2.imread(self.source_path) is not None:
                file_paths.append(self.source_path)
        return file_paths
    
    
    def get_filenames(self):
        if os.path.isdir(self.source_path):
            filenames = os.listdir(self.source_path)
        else:
            filenames = [os.path.basename(self.source_path)]
        return filenames
    
    
    def get_result_data(self):
        source = self.get_filepaths()
        obj = YOLOv8(self.task, self.subtask, self.model_path, source,
                    self.custom_class_colors, self.priority_classes,
                    self.plot, self.white_background, self.freq_table, 
                    self.freq_text_color, self.freq_table_color)
        return obj.results
    
    
    def get_des_path(self):
        des_path = self.des_dir
        if self.des_dir is None:
            if os.path.isdir(self.source_path):
                des_path = os.path.join(self.source_path, "results")
            else:
                des_path = os.path.join(os.path.dirname(self.source_path), 
                                        "results")
        os.makedirs(des_path, exist_ok=True)
        return des_path
    
    
    def copy_results(self):
        for img, filename in tqdm(zip(self.results, self.filenames), 
                                  desc='Copying results'):
            img_path = os.path.join(self.des_path, filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, img)
        
        
    def copy_cropped_results(self):
        for d, filename in tqdm(zip(self.results, self.filenames),
                                desc='Copying cropped results'):
            img_name, ext = os.path.splitext(filename)   
            if isinstance(d, dict):
                for cls, cropped_img_list in (d.items()):
                    cls_dir = os.path.join(self.des_path, cls)
                    os.makedirs(cls_dir, exist_ok=True)
                    for idx, cropped_img in enumerate(cropped_img_list):
                        name = f"{img_name}_({idx}){ext}"
                        img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                        img_path = os.path.join(cls_dir, name)
                        cv2.imwrite(img_path, img) 
    
    def save_pipeline(self):
        self.results = self.get_result_data()
        self.filenames = self.get_filenames()
        self.des_path = self.get_des_path()
        if self.task == self.subtask:
            self.copy_results()
        else:
            self.copy_cropped_results()
        print("\nDone.")

    
    def save_pipeline2(self):
        source = self.get_filepaths()
        self.des_path = self.get_des_path()
        filenames = os.listdir(self.source_path)
        for file_path, filename in zip(source, filenames):
            obj = YOLOv8(self.task, self.subtask, self.model_path, [file_path],
                        self.custom_class_colors, self.priority_classes,
                        self.plot, self.white_background, self.freq_table, 
                        self.freq_text_color, self.freq_table_color)
            r = obj.results 
            
            if self.task == self.subtask:
                img_path = os.path.join(self.des_path, filename)
                img = cv2.cvtColor(obj.results[0], cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_path, img)
            else:
                img_name, ext = os.path.splitext(filename) 
                for d in r:
                    if isinstance(d, dict):
                        for cls, cropped_img_list in (d.items()):
                            cls_dir = os.path.join(self.des_path, cls)
                            os.makedirs(cls_dir, exist_ok=True)
                            for idx, cropped_img in enumerate(cropped_img_list):
                                # name = f"{img_name}_({idx}){ext}"  
                                #png files use lossless compression
                                name = f"{img_name}_({idx}).png"                               
                                img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                                img_path = os.path.join(cls_dir, name)
                                cv2.imwrite(img_path, img) 

        print("\nDone.")
            
    
    
def main2(source_path, des_path, model_path, task='segment', subtask='crop'):
    for file in os.listdir(source_path):
        filepath = os.path.join(source_path, file)
        obj = copy_data(source_path=filepath, 
                        des_dir=des_path, 
                        task=task, subtask=subtask, 
                        model_path=model_path)
        obj.save_pipeline()
        
        
def main(source_path, des_path, model_path, species=None):
    if species is None: species=os.listdir(source_path)
    for folder in species:
        folder_path = os.path.join(source_path, folder)
        des_folder_path = os.path.join(des_path, folder)
        for type_folder in tqdm(os.listdir(folder_path)):
            type_folderpath = os.path.join(folder_path, type_folder)
            if not os.path.isdir(type_folderpath): 
                print("\nskipped {type_folderpath}\n")
                continue
            des_type_folderpath = os.path.join(des_folder_path, type_folder)
            obj = copy_data(source_path=type_folderpath, 
                            des_dir=des_type_folderpath, 
                            task='segment', subtask='crop', 
                            model_path=model_path)
            obj.save_pipeline2()
            
            
if __name__=='main':
    source_path = r"D:\New folder\qzense Dataset\Final Data"
    des_path = r"D:\New folder\Segmented Final Data"
    model_path=r"D:\New folder\Dataset versions\YOLO\Fish prawn segmentation\runs\segment\train4\weights\best.pt" 
    # species = os.listdir(source_path)[23:]
    species = ['Sardine']
    main(source_path, des_path, model_path, species)
    
    
    # from yolo_results import img_resizer
    # src_folder_path = r"D:\New folder\New app testing data-20231130T193006Z-001\New app testing data\input\2023-12-06\sardine\newly segmented\Sardine\Good\fish"
    # des_folder_path = r"D:\New folder\New app testing data-20231130T193006Z-001\New app testing data\input\2023-12-06\sardine\newly segmented\Sardine\Good\fish\resized"
    # target_size=(640, 640)
    # bg='black'
    # for file in tqdm(os.listdir(src_folder_path), desc='Loading '):
    #     os.makedirs(des_folder_path, exist_ok=True)
    #     filepath = os.path.join(src_folder_path, file)
    #     image = cv2.imread(filepath)
    #     if image is None: continue
    #     resizer = img_resizer(image, target_size, bg)
    #     resized_image = resizer.resized_image
    #     new_filepath = os.path.join(des_folder_path, file)
    #     cv2.imwrite(new_filepath, resized_image)
    
    
    
         