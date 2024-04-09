# coding: utf-8

import numpy as np
import os
import numpy as np
import scipy.ndimage
import json


def resize_img(img, img_shape, target_shape):
    # 计算缩放因子
    scale_depth = target_shape[0] / img_shape[0]
    scale_height = target_shape[1] / img_shape[1]
    scale_width = target_shape[2] / img_shape[2]

    # 使用scipy.ndimage.zoom进行重采样
    resized_img = scipy.ndimage.zoom(img, (scale_depth, scale_height, scale_width))
    return resized_img


if __name__ == '__main__':
    with open(r'max_shape.json', 'r') as f:
        dict_max_shape = json.load(f)
    path = r'/media/shucheng/MyBook/DL_dataset/M2ROI-FN'
    roi_ctx = [1006, 2006, 1015, 2015, 1035, 2035, 1032, 2032]
    roi_subctx = [17, 18, 53, 54]
    dict_ = {
        'T1': {},
        'Pet': {},
    }
    for roi in roi_ctx + roi_subctx:
        max_shape_roi = dict_max_shape[str(roi)]
        T1_ROI_list, Pet_ROI_list = [], []
        for subject_id in os.listdir(path):
            subject_path = os.path.join(path, subject_id)
            T1_ROI = np.load(os.path.join(subject_path, f'{roi}_T1.npy'))
            Pet_ROI = np.load(os.path.join(subject_path, f'{roi}_Pet.npy'))
            shape_image = T1_ROI.shape
            T1_ROI_resize = resize_img(T1_ROI, shape_image, max_shape_roi)
            Pet_ROI_resize = resize_img(Pet_ROI, shape_image, max_shape_roi)
            T1_ROI_list.append(T1_ROI_resize)
            Pet_ROI_list.append(Pet_ROI_resize)

        dict_['T1'][roi] = [np.array(T1_ROI_list).mean(), np.array(T1_ROI_list).std()]
        dict_['Pet'][roi] = [np.array(Pet_ROI_list).mean(), np.array(Pet_ROI_list).std()]

    with open(r'mean_std.json', 'w') as f1:
        json.dump(dict_, f1, indent=4)