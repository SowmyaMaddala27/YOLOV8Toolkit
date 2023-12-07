# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 23:21:32 2023

@author: sowmya
"""

import cv2

class freqTable:
    def __init__(self, freq_dict, image_arr, video_frame=False, all_frames_classes=None, 
                 text_color=(0, 0, 0), box_color=(0, 0, 0)):
        self.freq_dict = freq_dict
        self.image = image_arr
        self.height, self.width, channels = image_arr.shape
        self.video_frame = video_frame
        self.all_frames_classes = all_frames_classes
        self.text_color = text_color
        self.box_color = box_color
        self.c1_values = ['Class']+[str(key) for key in list(freq_dict.keys())]
        self.c2_values = ['Count']+[str(val) for val in list(freq_dict.values())]

    def get_col_info(self):
        class_names = self.freq_dict.keys()
        if self.video_frame: class_names = self.all_frames_classes
        max_width = len('Class')
        max_cls_len = max([len(cls) for cls in class_names])
        if len('Class')<=max_cls_len:
            max_width=max_cls_len
        self.col1_width = (max_width)*20
        self.col2_width = 20*(len('Count')+1)
        self.col1_height = len(class_names)+1

        return self.col1_width, self.col2_width, self.col1_height

    def draw_box(self, img):
        table_width = self.col1_width+self.col2_width
        table_height = self.col1_height*50

        self.dp1_x1 = self.width-table_width-10
        self.dp1_y1 = 30
        dp1 = (self.dp1_x1, self.dp1_y1)

        self.dp2_x1 = self.width-10
        self.dp2_y1 = 30+table_height
        dp2 = (self.dp2_x1, self.dp2_y1)

        cv2.rectangle(img, dp1, dp2, self.box_color, 2);

    def draw_rows(self, img):
        # draw horizontal lines
        p1_x1 = self.dp1_x1
        p1_y1 = self.dp1_y1
        p1 = (p1_x1, p1_y1)

        p2_x1 = self.dp2_x1
        p2_y1 = self.dp1_y1
        p2 = (p2_x1, p2_y1)

        for i in range(len(self.freq_dict)):
            p1_y1 = p1_y1+50
            p1 = (p1_x1, p1_y1)
            p2_y1 = p1_y1
            p2 = (p2_x1, p2_y1)
            cv2.line(img, p1, p2, self.box_color, 2, lineType=cv2.LINE_4);

    def draw_cols(self, img):
        p1_x1 = self.dp1_x1+self.col1_width
        p1_y1 = self.dp1_y1
        p1 = (p1_x1, p1_y1)

        p2_x1 = p1_x1
        p2_y1 = self.dp2_y1
        p2 = (p2_x1, p2_y1)

        cv2.line(img, p1, p2, self.box_color, 2, lineType=cv2.LINE_4);

    def put_class_data(self, img):
      # put column1(class) text
      p1_x1 = self.dp1_x1
      p1_y1 = self.dp1_y1

      for i in range(len(self.c1_values)):
          p1_y1 = p1_y1+50
          p1 = (p1_x1+10, p1_y1-10)
          cv2.putText(img, self.c1_values[i], p1, 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2);

    def put_counts_data(self, img):
        # put column2(count) text
        p1_x1 = self.dp1_x1+self.col1_width
        p1_y1 = self.dp1_y1
        p1 = (p1_x1, p1_y1)

        for i in range(len(self.c2_values)):
            p1_y1 = p1_y1+50
            p1 = (p1_x1+10, p1_y1-10)
            cv2.putText(img, self.c2_values[i], p1, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2);

    def plot_table(self):
        self.get_col_info()
        self.draw_box(self.image)
        self.draw_rows(self.image)
        self.draw_cols(self.image)
        self.put_class_data(self.image)
        self.put_counts_data(self.image)
        return self.image

