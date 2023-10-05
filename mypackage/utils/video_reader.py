# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 07:53:28 2023

@author: sowmya
"""

from cap_from_youtube import list_video_streams
from cap_from_youtube import cap_from_youtube
import cv2


def read_local_video(video_path):
    supported_formats = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov',
                         'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm']
    if video_path.split('.')[-1] not in supported_formats:
        print(f'Unsupported format: {video_path.split(".")[-1]}')
        return
    cap = cv2.VideoCapture(video_path)
    return cap

def read_youtube_video(video_url, resolution=None):
    if not resolution:
        cap = cap_from_youtube(video_url, resolution)
    else:
        cap = cap_from_youtube(video_url)
    return cap

def get_ytube_video_info(video_url):
    streams, resolutions = list_video_streams(video_url)
    for stream in streams:
        print(stream)
    return resolutions


def read_video(video_path, vid_pathtype):
    # Open the video file
    if vid_pathtype == 'local':
        cap = read_local_video(video_path)
    elif vid_pathtype == 'youtube':
        cap = read_youtube_video(video_path)

    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    return cap