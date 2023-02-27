#this code is used for visialize the VGG label:
import numpy as np
import skimage
import json
from skimage import io
from skimage import draw

import matplotlib.pyplot as plt

def ReadJson(jsonfile):
    with open(jsonfile,encoding='utf-8') as f:
        jsonData = json.load(f)
    return jsonData
#read the json file
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

def test_visial(demo_name, image_folder, path_annotation_json):

    annotations = ReadJson(path_annotation_json)
    data = return_jpg_json(annotations, demo_name)
    regions = data["regions"]

    img = read_img(image_folder, demo_name)
    # io.imshow(img)
    shape =img.shape
    mask = np.zeros((shape[0], shape[1]))
    for bas in regions:
        points_x = regions[bas]['shape_attributes']["all_points_x"]
        points_y = regions[bas]['shape_attributes']["all_points_y"]
        rr, cc = draw_label(points_y, points_x)
        try:
            generatemask(mask, rr, cc)
        except:
            generatemask(mask, rr, cc)

    mask_com = np.array([mask]*3)
    mask_com = mask_com.transpose(1, 2, 0)

    splash = visal(img, mask_com)
    io.imshow(splash)
    plt.show()
#main fuction for sample image

if __name__ == '__main__':
    demo_name = "LA_2_80_edemo_0.jpg" # sample image
    image_folder = "F:\\expan\\" #image path
    path_annotation_json = 'F:\\expan\\expan_via_region_data_0.json' # .json

    test_visial()