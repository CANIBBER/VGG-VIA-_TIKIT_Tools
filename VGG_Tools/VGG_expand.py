#this code is used for expand images' scale and generate new VGG labels

import imageio
import skimage
from skimage import io
from skimage import draw
import cv2
import json
import numpy as np
import random

from math import cos, sin, pi, fabs, radians #内置数学类函数库

import matplotlib.pyplot as plt

def generate_bounding(x_limit, y_limit):
    up = random.randint(100, x_limit//2)
    down = random.randint(100, x_limit-up)

    left = random.randint(100, y_limit//2)
    right = random.randint(100, y_limit-left)
    return up, down, left, right
# generate bounding pixels for new image

def expan_image(img, up, down, left, right):
    img1 = cv2.copyMakeBorder(img, up, down, left, right, cv2.BORDER_CONSTANT, value=[100, 100, 100])
    return img1
# expand image

def ReadJson(jsonfile):
    with open(jsonfile, encoding='utf-8') as f:
        jsonData = json.load(f)
    return jsonData
# read json data

def return_jpg_json(annotations, Demoname):
    for key in annotations:
        data = annotations[key]
        if Demoname == data["filename"]:
            return data
            break
#return the labels for target image

def read_img(image_folder, demoname):
    path = image_folder + demoname
    img = io.imread(path)
    return img
#read image

def draw_label(allx, ally):
    rr, cc = draw.polygon(allx, ally)
    return rr, cc
# draw polygon of labels

def generatemask(mask, rr, cc):
    mask[rr, cc] = 1
# generate the mask

def visal(img, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(img)) * 125

    splash = np.where(mask, img, gray).astype(np.uint8)
    return splash
# splash the image with labels


def test_expan():
    demo_name = "LA_1_3.jpg" # sample image
    image_folder = "D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\"  # original image file
    path_annotation_json = 'D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\via_region_data.json'  # .json
    img = read_img(image_folder, demo_name)

    x_limit = random.randint(400, 1200)
    y_limit = random.randint(400, 1200)
    up, down, left, right = generate_bounding(x_limit, y_limit)
    img_ex = expan_image(img, up, down, left, right)

    mask = np.zeros((img_ex.shape[0], img_ex.shape[1]), dtype=np.uint8)
    annotations = ReadJson(path_annotation_json)
    data = return_jpg_json(annotations, demo_name)
    regions = data["regions"]
    height, width = img.shape[:2]
    cy, cx = height // 2, width // 2
    for bas in regions:
        points_x = regions[bas]['shape_attributes']["all_points_x"]
        points_y = regions[bas]['shape_attributes']["all_points_y"]

        new_x = []
        new_y = []
        for x in points_x:
            new_x.append(x + left)
        for y in points_y:
            new_y.append(y + up)
        rr, cc = draw_label(new_y, new_x)
        generatemask(mask, rr, cc)
    mask_com = np.array([mask] * 3)
    mask_com = mask_com.transpose(1, 2, 0)

    splash = visal(img_ex, mask_com)
    io.imshow(splash)
    plt.show()
# sample function and visualize

def expan_and_save(image_folder, path_annotation_json, path_save_json, path_save, post):
    annotations = json.load(open(path_annotation_json, 'r'))
    Alllabels = {}
    for key in annotations:
        data = annotations[key]
        name = data["filename"]

        angle = random.randint(0, 180)
        # print(angle)

        img = read_img(image_folder, name)
        # ro, ra = RotateImage(img, angle)
        #
        # new_cy = ro.shape[1] // 2
        # new_cx = ro.shape[0] // 2
        x_limit = random.randint(600, 1300)
        y_limit = random.randint(500, 1300)
        up, down, left, right = generate_bounding(x_limit, y_limit)
        img_ex = expan_image(img, up, down, left, right)

        mask = np.zeros((img_ex.shape[0], img_ex.shape[1]), dtype=np.uint8)
        area = img_ex.shape[1] * img_ex.shape[0]

        data = return_jpg_json(annotations, name)
        regions = data["regions"]
        height, width = img.shape[:2]
        cy, cx = height // 2, width // 2
        imageio.imwrite((path_save + name.replace(".jpg", post)), img_ex)

        A = {}
        for bas in regions:
            points_x = regions[bas]['shape_attributes']["all_points_x"]
            points_y = regions[bas]['shape_attributes']["all_points_y"]

            new_x = []
            new_y = []
            for x in points_x:
                new_x.append(x + left)
            for y in points_y:
                new_y.append(y + up)

            A[str(bas)] = {"shape_attributes": {'all_points_x': new_x, 'all_points_y': new_y}, "region_attributes": {}}
        # print(A)
        B = {"fileref": "", "size": str(area), "filename": name.replace(".jpg", post), "base64_img_data": "",
             "file_attributes": {}, "regions": A}
        Alllabels[name.replace(".jpg", post) + str(area)] = B
    # print(Alllabels)
    Alllabels = json.dumps(Alllabels).replace("\\", "").replace("\"{", "{").replace("}\"", "}")
    f = open(path_save_json, "w")
    f.write(Alllabels)
    f.close()

# batch processing function

if __name__ == "__main__":

    image_folder = "D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\"  # original image file path
    path_annotation_json = 'D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\via_region_data.json'  # original .json path

    path_save_json = "F:\\expan\\expan_via_region_data_0.json"  # new.json path
    path_save = "F:\\expan\\"  # new image save path
    post = "_edemo_0.jpg" # rename image

    expan_and_save(image_folder, path_annotation_json, path_save_json, path_save, post)