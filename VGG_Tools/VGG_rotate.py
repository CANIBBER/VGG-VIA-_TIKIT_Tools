#this code is used for visialize the VGG label:


import skimage
import imageio
from skimage import io
from skimage import draw
import cv2
import json
import numpy as np
import random
from shapely.geometry import Polygon
from math import cos, sin, pi, fabs, radians

import matplotlib.pyplot as plt

def ReadJson(jsonfile):
    with open(jsonfile, encoding='utf-8') as f:
        jsonData = json.load(f)
    return jsonData

def return_jpg_json(annotations, Demoname):
    for key in annotations:
        data = annotations[key]
        if Demoname == data["filename"]:
            return data
            break

def read_img(image_folder, demoname):
    path = image_folder + demoname
    img = io.imread(path)
    return img

def draw_label(allx, ally):
    rr, cc = draw.polygon(allx, ally)
    return rr, cc

def generatemask(mask, rr, cc):
    mask[rr, cc] = 1


def visal(img, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(img)) * 125

    splash = np.where(mask, img, gray).astype(np.uint8)
    return splash

def RotateImage(img, degree):
    height, width = img.shape[:2]    #获得图片的高和宽
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), -degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    # print(width // 2, height // 2)
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(100, 100, 100))
    return imgRotation, matRotation
#rotate image


# images_path = 'D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\'  #图片的根目录
# save_path = "D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\gene\\" #保存图片文件夹
# after = "_demoa1.jpg"
# file_list = os.listdir(images_path)
# for img_name in file_list:
#     print(img_name)

def rotate_xy(x, y, angle, cx, cy, new_cx, new_cy):
    # print(cx,cy)
    angle = angle * pi / 180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + new_cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + new_cy
    return x_new, y_new
# counting new coordinate of point

def test_rota(angle):
    demo_name = "LA_1_3.jpg"
    image_folder = "D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\"  # 样本所在文件夹
    path_annotation_json = 'D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\via_region_data.json'  # .json文件

    img = read_img(image_folder, demo_name)
    ro, ra = RotateImage(img, angle)
    # io.imshow(ro)
    # plt.show()
    new_cy = ro.shape[1]//2
    new_cx = ro.shape[0]//2
    mask = np.zeros((ro.shape[0], ro.shape[1]), dtype=np.uint8)
    annotations = ReadJson(path_annotation_json)
    data = return_jpg_json(annotations, demo_name)
    regions = data["regions"]
    height, width = img.shape[:2]
    cy, cx = height//2, width//2
    for bas in regions:
        points_x = regions[bas]['shape_attributes']["all_points_x"]
        points_y = regions[bas]['shape_attributes']["all_points_y"]

        new_x = []
        new_y = []
        # points = []
        # for x in points_x:
        #     for y in points_y:
        #         points.append((x, y))
                # nx, ny = rotate_xy(x, y, angle, cx, cy)
                # if nx not in new_x:
                #     new_x.append(nx)
                # if ny not in new_y:
                #     new_y.append(ny)

        some_poly = Polygon(zip(points_x, points_y))
        BOUNDING = list(some_poly.exterior.coords)[:-1]

        for point in BOUNDING:
            nx, ny = rotate_xy(point[0], point[1], angle, cx, cy, new_cx, new_cy)
            new_x.append(int(nx))
            new_y.append(int(ny))
            rr, cc = draw_label(new_y, new_x)
            generatemask(mask, rr, cc)
    mask_com = np.array([mask] * 3)
    mask_com = mask_com.transpose(1, 2, 0)

    splash = visal(ro, mask_com)
    io.imshow(splash)
    plt.show()
        # print(points_x)
        # print(points_y)
        # print(new_x)
        # print(new_y)
        # print("--")
# sample function and visualize

def rotate_and_save(image_folder, path_annotation_json, path_save_json, path_save, post):
    annotations = json.load(open(path_annotation_json, 'r'))
    Alllabels = {}
    for key in annotations:
        data = annotations[key]
        name = data["filename"]

        angle = random.randint(0, 180)
        # print(angle)

        img = read_img(image_folder, name)
        ro, ra = RotateImage(img, angle)

        new_cy = ro.shape[1] // 2
        new_cx = ro.shape[0] // 2
        area = ro.shape[1] * ro.shape[0]

        data = return_jpg_json(annotations, name)
        regions = data["regions"]
        height, width = img.shape[:2]
        cy, cx = height // 2, width // 2
        imageio.imwrite((path_save + name.replace(".jpg", post)), ro)

        A = {}
        for bas in regions:
            points_x = regions[bas]['shape_attributes']["all_points_x"]
            points_y = regions[bas]['shape_attributes']["all_points_y"]

            new_x = []
            new_y = []

            some_poly = Polygon(zip(points_x, points_y))
            BOUNDING = list(some_poly.exterior.coords)[:-1]

            for point in BOUNDING:
                nx, ny = rotate_xy(point[0], point[1], angle, cx, cy, new_cx, new_cy)
                if nx > width:
                    nx = width
                if ny > height:
                    ny = height
                new_x.append(int(nx))
                new_y.append(int(ny))
                # print(str(regions))
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

if __name__ == '__main__':
    # test_rota(45)
    image_folder = "D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\"  # images
    path_annotation_json = 'D:\\combined_program\\Mask_RCNN\\samples\\balloon\\new_logistics\\train\\via_region_data.json'  # .json
    path_save_json = "F:\\rota\\via_region_data_4.json"#save new .json
    path_save = "F:\\rota\\"# save path
    post = "_demo_5.jpg"# rename image

    rotate_and_save(image_folder, path_annotation_json, path_save_json, path_save, post)