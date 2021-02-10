import numpy as np
import cv2
import pandas as pd
# import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from random import randint

r = 1  # scale down
width = 256  # patch size
batch_size = 100
sigma_elephant = 10
elephant_crop = 0
background_crop = 0


FOLDERPATH = r"D:\Dataset\AED"
DEBUG = True


def read_ignore_list():
    df_ignore = pd.read_csv(FOLDERPATH + '/MismatchedTrainImages.txt')
    ignore_list = df_ignore['train_id'].tolist()
    return ignore_list


def read_coordinates(img_id):
    df_coordinates = pd.read_csv(FOLDERPATH + '/training_elephants.csv')
    img_df = df_coordinates[df_coordinates['tid'] == os.path.splitext(img_id)[0]]
    if DEBUG:
        print("Function: read_coordinates() {}".format(len(img_df)))
    x_coord = img_df['row'].tolist()
    y_coord = img_df['col'].tolist()
    return x_coord, y_coord


def gen_data(image_list):
    ignore_list = read_ignore_list()
    iter_count = 0
    for image in image_list:
        if image in ignore_list:
            print('  Ignored...{}'.format(image))
            continue
        if DEBUG:
            print("Reading {}".format(image))

        elephant_image = cv2.imread(FOLDERPATH + '/training_images/' + image)
        elephant_mask = np.zeros((elephant_image.shape[0], elephant_image.shape[1]), dtype=np.float32)
        if DEBUG:
            print("Image Shape {}".format(elephant_image.shape))

        x_coord, y_coord = read_coordinates(image)
        for x, y in zip(x_coord, y_coord):
            elephant_mask[y, x] = 1
        if DEBUG:
            print("Before Density Count", elephant_mask.sum())

        elephant_mask[:, :] = gaussian_filter(elephant_mask[:, :], sigma=sigma_elephant)
        if DEBUG:
            print("After Density Count", elephant_mask.sum())

        # slidingwindow algorithm
        elephant, background = sliding_window_crop(elephant_image, elephant_mask, image)
        iter_count += 1
        print(".....Image {}/{} cropping done".format(iter_count, len(image_list)))

    print("#####################################################")
    print("Dataset Summary")
    print("#####################################################")
    print("Total no.of Images:              {}".format(len(image_list)))
    print("No.of patches with elephant:     {}".format(elephant))
    print("No.of patches without elephant:  {}".format(background))
    print("#####################################################")


def sliding_window_crop(elephant_image, elephant_mask, image_id):
    width = height = 256
    stride = 0
    background_count = 2
    global elephant_crop, background_crop

    factor_x = elephant_image.shape[0] / 256
    factor_y = elephant_image.shape[1] / 256
    fact_new_x = int(height - (height * (factor_x % 1))) if factor_x != 0 else 0
    fact_new_y = int(width - (width * (factor_y % 1))) if factor_y != 0 else 0
    image_resize = cv2.copyMakeBorder(elephant_image, 0, fact_new_x, 0, fact_new_y, cv2.BORDER_CONSTANT)
    mask_resize = cv2.copyMakeBorder(elephant_mask, 0, fact_new_x, 0, fact_new_y, cv2.BORDER_CONSTANT)
    if DEBUG:
        print("Before resize: Image {} and Mask {} ---> After resize: Image {} and Mask {}".format(elephant_image.shape,
                                                                                                   elephant_mask.shape,
                                                                                                   image_resize.shape,
                                                                                                   mask_resize.shape))

    for i in range(0, int(image_resize.shape[0] / 256)):
        for j in range(0, int(image_resize.shape[1] / 256)):
            if width * i - stride > 0:
                xstart = width * i - stride
            else:
                xstart = 0

            if width * j - stride > 0:
                ystart = width * j - stride
            else:
                ystart = 0

            xstop = width + width * i
            ystop = height + height * j

            image_crop = image_resize[xstart:xstop, ystart:ystop, :]
            mask_crop = mask_resize[xstart:xstop, ystart:ystop]
            if background_count != 0 and mask_crop.sum() == 0:
                background_count -= 1
                mask_crop_bg = np.zeros((width,height))
                np.save('./Data/Train_Data/Images_BG/' + str(int((xstart + xstop) / 2)) + '_' + str(
                    int((ystart + ystop) / 2)) + '_' + os.path.splitext(image_id)[0], image_crop)
                np.save('./Data/Train_Data/Masks_BG/' + str(int((xstart + xstop) / 2)) + '_' + str(
                    int((ystart + ystop) / 2)) + '_' + os.path.splitext(image_id)[0], mask_crop)
                background_crop += 1

            if mask_crop.sum() > 0:
                np.save('./Data/Train_Data/Images/' + str(int((xstart + xstop) / 2)) + '_' + str(
                    int((ystart + ystop) / 2)) + '_' + os.path.splitext(image_id)[0], image_crop)
                np.save('./Data/Train_Data/Masks/' + str(int((xstart + xstop) / 2)) + '_' + str(
                    int((ystart + ystop) / 2)) + '_' + os.path.splitext(image_id)[0], mask_crop)
                elephant_crop += 1

    return elephant_crop, background_crop


if __name__ == '__main__':
    image_list = os.listdir(FOLDERPATH + '/training_images')
    # print(image_list)
    gen_data(image_list)