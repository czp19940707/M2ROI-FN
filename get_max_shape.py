import os
import numpy as np
import json

# roi_ctx = [1006, 2006, 1015, 2015, 1035, 2035, ]
# roi_subctx = [17, 18, 53, 54]

roi_ctx = [1032, 2032]
roi_subctx = []


def get_max_shape():
    pass


if __name__ == '__main__':
    dict_ = {}
    path = r'/media/shucheng/MyBook/DL_dataset/M2ROI-FN'
    for roi in roi_ctx + roi_subctx:
        roi_shape_list = []
        for subject_id in os.listdir(path):
            data_shape = np.load(os.path.join(path, subject_id, f'{roi}_Mask.npy')).shape
            roi_shape_list.append(data_shape)
        roi_shape_list_max = np.array(roi_shape_list).max(0).tolist()
        dict_[roi] = roi_shape_list_max

    with open(r'max_shape1.json', 'w') as f:
        json.dump(dict_, f, indent=4)
