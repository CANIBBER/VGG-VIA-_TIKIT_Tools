#This code is used to transfer labels into mask png for Semantic Segmentation
import os
import json
import numpy as np
import skimage.draw
import cv2
import skimage
from skimage import io

def main():

    image_folder = "D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\"  # original image file
    mask_folder = "D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\mask\\"  # mask save file
    path_annotation_json = 'D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\via_region_data.json'  # .json

    annotations = json.load(open(path_annotation_json, 'r'))
    for key in annotations:
        data = annotations[key]
        name = data["filename"]
        img = io.imread(image_folder + name)
        maskImage = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        # maskImage = maskImage*255
        regions = data["regions"]
        for bas in regions:
            points_x = regions[bas]['shape_attributes']["all_points_x"]
            points_y = regions[bas]['shape_attributes']["all_points_y"]
            rr, cc = skimage.draw.polygon(points_y, points_x)
            maskImage[rr, cc] = 255
            # skimage.io.imshow(maskImage)
        save = mask_folder + name.replace("jpg", "png")
        io.imsave(save, maskImage)


