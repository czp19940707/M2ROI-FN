import os
import pandas
import pandas as pd
import numpy as np
import json

dict_label = {
    'lh-middletemporal': ['aparc', 'lh-middletemporal', 'lh', 15, 1000],
    'lh-insula': ['aparc', 'lh-insula', 'lh', 35, 1000],
    'lh-entorhinal': ['aparc', 'lh-entorhinal', 'lh', 6, 1000],
    'rh-middletemporal': ['aparc', 'rh-middletemporal', 'rh', 15, 2000],
    'rh-insula': ['aparc', 'rh-insula', 'rh', 35, 2000],
    'rh-entorhinal': ['aparc', 'rh-entorhinal', 'rh', 6, 2000],
    'Left-Hippocampus': ['aseg', 'Left-Hippocampus', 'lh', 17, 0],
    'Left-Amygdala': ['aseg', 'Left-Amygdala', 'lh', 18, 0],
    'Right-Hippocampus': ['aseg', 'Right-Hippocampus', 'rh', 53, 0],
    'Right-Amygdala': ['aseg', 'Right-Amygdala', 'rh', 54, 0],
}

dict_max_shape = {
    'lh-middletemporal': [48, 68, 113],
    'lh-insula': [47, 53, 72],
    'lh-entorhinal': [29, 30, 40],
    'rh-middletemporal': [49, 66, 110],
    'rh-insula': [47, 51, 69],
    'rh-entorhinal': [31, 29, 42],
    'Left-Hippocampus': [33, 37, 51],
    'Left-Amygdala': [25, 23, 24],
    'Right-Hippocampus': [35, 36, 50],
    'Right-Amygdala': [26, 24, 24],
}

dict_network = {
    'lh-middletemporal': [[3, 5, 8], 4],
    'lh-insula': [[3, 4, 5], 4],
    'lh-entorhinal': [[2, 2, 3], 4],
    'rh-middletemporal': [[4, 5, 7], 4],
    'rh-insula': [[3, 4, 5], 4],
    'rh-entorhinal': [[2, 2, 3], 4],
    'Left-Hippocampus': [[3, 3, 4], 1],
    'Left-Amygdala': [[2, 2, 2], 1],
    'Right-Hippocampus': [[3, 3, 4], 1],
    'Right-Amygdala': [[2, 2, 2], 1],
}

# 'Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'lh-insula', 'rh-insula',
#                  'lh-middletemporal', 'rh-middletemporal', 'lh-entorhinal', 'rh-entorhinal'

dict_train = {
    'v4': {
        'rois': [1015, 2015],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v3': {
        'rois': [1035, 2035],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v5': {
        'rois': [1006, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v1': {
        'rois': [17, 53],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v2': {
        'rois': [18, 54],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v6': {
        'rois': [17, 53, 18, 54],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v7': {
        'rois': [17, 53, 18, 54, 1035, 2035],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v8': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v9': {
        'rois': [53, 54, 2035, 2015, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v10': {
        'rois': [17, 18, 1035, 1015, 1006],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v11': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'T1',
    },
    'v12': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'pet',
    },
    'v13': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v14': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': True,
        'morphological': True,
        'modality': 'T1_pet',
    },
    'v15': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': False,
        'morphological': True,
        'modality': 'T1_pet',
    },
    'v1-s': {
            'rois': [1015, 2015],
            'mask': False,
            'morphological': False,
            'modality': 'T1_pet',
        },
    'v2-s': {
        'rois': [1035, 2035],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v3-s': {
        'rois': [1006, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v4-s': {
        'rois': [17, 53],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v5-s': {
        'rois': [18, 54],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v6-s': {
        'rois': [17, 53, 18, 54],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v7-s': {
        'rois': [17, 53, 18, 54, 1035, 2035],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v8-s': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v9-s': {
        'rois': [53, 54, 2035, 2015, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v10-s': {
        'rois': [17, 18, 1035, 1015, 1006],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v11-s': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'T1',
    },
    'v12-s': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'pet',
    },
    'v13-s': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
    'v14-s': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': True,
        'morphological': True,
        'modality': 'T1_pet',
    },
    'v15-s': {
        'rois': [17, 53, 18, 54, 1035, 2035, 1015, 2015, 1006, 2006],
        'mask': False,
        'morphological': True,
        'modality': 'T1_pet',
    },

    # 20240401
    'v16': {
        'rois': [1032, 2032],
        'mask': False,
        'morphological': False,
        'modality': 'pet',
    },
    'v17': {
        'rois': [17, 53],
        'mask': False,
        'morphological': False,
        'modality': 'pet',
    },
    'v18': {
        'rois': [18, 54],
        'mask': False,
        'morphological': False,
        'modality': 'pet',
    }

    # 'z6': {
    #     'rois': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala'],
    #     'mask': False,
    #     'morphological': False,
    #     'modality': 'T1_pet',
    # },
    # 'z7': {
    #     'rois': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'lh-insula', 'rh-insula'],
    #     'mask': False,
    #     'morphological': False,
    #     'modality': 'T1_pet',
    # },
    # 'z8': {
    #     'rois': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'lh-insula', 'rh-insula',
    #              'lh-middletemporal', 'rh-middletemporal'],
    #     'mask': False,
    #     'morphological': False,
    #     'modality': 'T1_pet',
    # },
    # 'z9': {
    #     'rois': ['Right-Hippocampus', 'Right-Amygdala', 'rh-insula', 'rh-middletemporal', 'rh-entorhinal'],
    #     'mask': False,
    #     'morphological': False,
    #     'modality': 'T1_pet',
    # },
    # 'z10': {
    #     'rois': ['Left-Hippocampus', 'Left-Amygdala', 'lh-insula', 'lh-middletemporal', 'lh-entorhinal'],
    #     'mask': False,
    #     'morphological': False,
    #     'modality': 'T1_pet',
    # },
    # 'z11': {
    #     'rois': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'lh-insula', 'rh-insula',
    #              'lh-middletemporal', 'rh-middletemporal', 'lh-entorhinal', 'rh-entorhinal'],
    #     'mask': False,
    #     'morphological': False,
    #     'modality': 'T1',
    # },
    # 'z12': {
    #     'rois': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'lh-insula', 'rh-insula',
    #              'lh-middletemporal', 'rh-middletemporal', 'lh-entorhinal', 'rh-entorhinal'],
    #     'mask': False,
    #     'morphological': False,
    #     'modality': 'pet',
    # },
    # 'z13': {
    #     'rois': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'lh-insula', 'rh-insula',
    #              'lh-middletemporal', 'rh-middletemporal', 'lh-entorhinal', 'rh-entorhinal'],
    #     'mask': False,
    #     'morphological': False,
    #     'modality': 'T1_pet',
    # },
    # 'z14': {
    #     'rois': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'lh-insula', 'rh-insula',
    #              'lh-middletemporal', 'rh-middletemporal', 'lh-entorhinal', 'rh-entorhinal'],
    #     'mask': True,
    #     'morphological': True,
    #     'modality': 'T1_pet',
    # },
    # 'z15': {
    #     'rois': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'lh-insula', 'rh-insula',
    #              'lh-middletemporal', 'rh-middletemporal', 'lh-entorhinal', 'rh-entorhinal'],
    #     'mask': False,
    #     'morphological': True,
    #     'modality': 'T1_pet',
    # },
}

dict_train_1 = {
    's1': {
        'rois': ['lh-middletemporal', 'rh-middletemporal'],
        'mask': True,
        'morphological': False,
        'modality': 'T1_pet',
    },
    's2': {
        'rois': ['lh-insula', 'rh-insula'],
        'mask': True,
        'morphological': False,
        'modality': 'T1_pet',
    },
    's3': {
        'rois': ['lh-entorhinal', 'rh-entorhinal'],
        'mask': True,
        'morphological': False,
        'modality': 'T1_pet',
    },
    's4': {
        'rois': ['Left-Hippocampus', 'Right-Hippocampus'],
        'mask': True,
        'morphological': False,
        'modality': 'T1_pet',
    },
    's5': {
        'rois': ['Left-Amygdala', 'Right-Amygdala'],
        'mask': True,
        'morphological': False,
        'modality': 'T1_pet',
    },
}

dict_train_2 = {
    'z6': {
        'rois': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala'],
        'mask': False,
        'morphological': False,
        'modality': 'T1_pet',
    },
}
dict_data_table_columns = {
    '1006': ['lh.Meanpial_lgi.aparc.entorhinal', 'lh.ThickAvg.aparc.entorhinal', 'lh.SurfArea.aparc.entorhinal',
             'lh.GausCurv.aparc.entorhinal', 'lh.GrayVol.aparc.entorhinal'],
    '2006': ['rh.Meanpial_lgi.aparc.entorhinal', 'rh.ThickAvg.aparc.entorhinal', 'rh.SurfArea.aparc.entorhinal',
             'rh.GausCurv.aparc.entorhinal', 'rh.GrayVol.aparc.entorhinal'],
    '1015': ['lh.Meanpial_lgi.aparc.middletemporal', 'lh.ThickAvg.aparc.middletemporal',
             'lh.SurfArea.aparc.middletemporal',
             'lh.GausCurv.aparc.middletemporal', 'lh.GrayVol.aparc.middletemporal'],
    '2015': ['rh.Meanpial_lgi.aparc.middletemporal', 'rh.ThickAvg.aparc.middletemporal',
             'rh.SurfArea.aparc.middletemporal',
             'rh.GausCurv.aparc.middletemporal', 'rh.GrayVol.aparc.middletemporal'],
    '1035': ['lh.Meanpial_lgi.aparc.insula', 'lh.ThickAvg.aparc.insula', 'lh.SurfArea.aparc.insula',
             'lh.GausCurv.aparc.insula', 'lh.GrayVol.aparc.insula'],
    '2035': ['rh.Meanpial_lgi.aparc.insula', 'rh.ThickAvg.aparc.insula', 'rh.SurfArea.aparc.insula',
             'rh.GausCurv.aparc.insula', 'rh.GrayVol.aparc.insula'],
    '1032': ['lh.Meanpial_lgi.aparc.frontalpole', 'lh.ThickAvg.aparc.frontalpole', 'lh.SurfArea.aparc.frontalpole',
             'lh.GausCurv.aparc.frontalpole', 'lh.GrayVol.aparc.frontalpole'],
    '2032': ['rh.Meanpial_lgi.aparc.frontalpole', 'rh.ThickAvg.aparc.frontalpole', 'rh.SurfArea.aparc.frontalpole',
             'rh.GausCurv.aparc.frontalpole', 'rh.GrayVol.aparc.frontalpole'],
    '17': ['Volume_mm3.aseg.Left-Hippocampus', 'Mean.aseg.Left-Hippocampus'],
    '18': ['Volume_mm3.aseg.Left-Amygdala', 'Mean.aseg.Left-Amygdala'],
    '53': ['Volume_mm3.aseg.Right-Hippocampus', 'Mean.aseg.Right-Hippocampus'],
    '54': ['Volume_mm3.aseg.Right-Amygdala', 'Mean.aseg.Right-Amygdala'],

}


def get_columns():
    list_ = []
    for key in dict_data_table_columns.keys():
        for column in dict_data_table_columns[key]:
            list_.append(column)
    return list_


def model_select(args):
    if args.v.endswith('-s'):
        from model import M2ROI_FN
        return M2ROI_FN(args.v)
    else:
        from model import M2ROI_FN_orig
        return M2ROI_FN_orig(args.v)
