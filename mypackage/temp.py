import numpy as np
import cv2
from ultralytics import YOLO
import os

def extract_single_image_segment(img, mask_segment):
    height, width = img.shape[0], img.shape[1]
    segment = np.array(mask_segment, dtype=np.int32)
    segment = segment.reshape((-1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [segment], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(segment)
    segment_image = np.zeros((h, w, 3), dtype=np.uint8)
    segment_image[0:h, 0:w] = masked_img[y:y+h, x:x+w]
    return segment_image

def segment_img(img, mask_points):
    height, width = img.shape[0], img.shape[1]
    mask_points = mask_points.astype(int)
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(binary_mask, [mask_points], 255)
    masked_img = cv2.bitwise_and(img, img,
                                  mask=binary_mask)
    x, y, w, h = cv2.boundingRect(mask_points)
    segment_image = np.zeros((height, width, 3), dtype=np.uint8)
    segment_image[y:y+h, x:x+w] = masked_img[y:y+h, x:x+w]
    segment_image = segment_image[y:y+h, x:x+w] 
    return segment_image


model_path = r"D:\New folder\Fish-Data-Science-project-Jupyter-notebooks\New Experiments\fish_freshness_classification - Copy\saved_models\segmentation\best (1)_fish.pt"
model = YOLO(model_path)
source = r"D:\New folder\New app testing data\input\2023-11-23\sardine\Good\2023-11-23_09_45_29_(2193)_sardine_input.jpeg"
des_folderpath = r"D:\New folder\New app testing data(2)\sardine input\2023-11-23\sardine\newly_segmented\completely new\my func"

def main():
    result = model(source, conf=0.75, iou=0.7)[0]
    image = result.orig_img
    mask = result.masks.xy    
    filepath = os.path.join(des_folderpath, "fish.png")
    if len(mask)==0: 
        cv2.imwrite(filepath, image)
    else: 
        cropped_images = []
        # Process results list
        for i in mask:
          # Masks object for segmentation masks outputs
            # rgb_crop = extract_single_image_segment(image, i)
            rgb_crop = segment_img(image, i)
            cropped_images.append(rgb_crop)
            filepath = os.path.join(des_folderpath, f"fish({len(cropped_images)}).png")
            cv2.imwrite(filepath, rgb_crop)
