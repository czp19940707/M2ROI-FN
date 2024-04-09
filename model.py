import numpy as np
from torch import nn
import torch
import pandas as pd
import math
from torchvision.models import resnet50
import os
from utils import dict_train


class PadMaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        super(PadMaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size, stride, return_indices=return_indices)
        self.pad = nn.ConstantPad3d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [self.stride - f_maps.size(i + 2) % self.stride for i in range(3)]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0

        self.pad.padding = (coords[2], 0, coords[1], 0, coords[0], 0)

        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, indices, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output, indices

        else:
            output = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv = nn.Sequential(
            # block1
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            # block2
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            # block3
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            # block4
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

    def forward(self, x):
        return self.conv(x)


class M2ROI_FN(nn.Module):
    def __init__(self, version='v1'):
        """
        :param modality: str 'pet or T1 or T1_pet'
        """
        super(M2ROI_FN, self).__init__()
        modality_list = [i for i in dict_train[version]['modality'].split('_')]
        rois_list = dict_train[version]['rois']
        morphological = dict_train[version]['morphological']
        self.cnn_dict = nn.ModuleDict()
        dict_network = self.load_dict_network()
        for m in modality_list:
            self.cnn_dict[m] = nn.ModuleDict()
            self.cnn_dict[m]['conv'] = BaseModel().conv     # basemodel
            for roi in rois_list:
                super_para_last_conv = dict_network[roi][0]
                self.cnn_dict[m][str(roi)] = nn.Conv3d(64, 64, tuple(super_para_last_conv))     # ROI conv

        if len(modality_list) > 1:
            self.mni = nn.ModuleDict()
            for roi in rois_list:
                self.mni[str(roi)] = nn.Sequential(
                    nn.Linear(64 * len(modality_list), 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                )
        if morphological:
            morphological_dim = np.sum(np.array([dict_network[i][-1] for i in rois_list]))
        else:
            morphological_dim = 0
        self.ROI_based_MLP_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * len(rois_list) + morphological_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        self.modality_list = modality_list
        self.rois_list = rois_list
        self.morphological = morphological
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_dict_network(self):
        import json
        from utils import dict_data_table_columns
        with open(r'conv_kernal_shape.json', 'r') as f:
            dict_1 = json.load(f)
        dict_1_new_version = {}
        for key in dict_1.keys():
            dict_1_new_version[int(key)] = [dict_1[key], len(dict_data_table_columns[key])]
        return dict_1_new_version

    def forward(self, x):
        features_dict = {}
        for m in self.modality_list:
            features_dict[m] = {}
            for roi in self.rois_list:
                roi = str(roi)      # change in 20240323
                low_level_local_features = self.cnn_dict[m]['conv'](x[roi][m].float().to(self.device))
                low_level_local_features = nn.Flatten()(self.cnn_dict[m][roi](low_level_local_features))
                features_dict[m][roi] = low_level_local_features

        features_dict['high_level'] = {}
        if len(self.modality_list) > 1:
            for roi in self.rois_list:
                roi = str(roi)  # change in 20240323
                temp_features = torch.cat(
                    [features_dict[self.modality_list[0]][roi],
                     features_dict[self.modality_list[1]][roi],
                     ], dim=1
                )
                features_dict['high_level'][roi] = self.mni[roi](temp_features)
        else:
            features_dict['high_level'] = features_dict[self.modality_list[0]]
        if self.morphological:
            features_dict['morphological'] = []
            for roi in self.rois_list:
                roi = str(roi)
                features_dict['morphological'].append(x[roi]['mor'])
            features_dict['morphological'] = torch.cat(features_dict['morphological'], dim=1)
        else:
            features_dict['morphological'] = torch.tensor([])
        global_features = torch.cat([features_dict['high_level'][i] for i in features_dict['high_level'].keys()] + [
            features_dict['morphological'].float().to(self.device)], dim=1)
        return self.ROI_based_MLP_classifier(global_features)


class M2ROI_FN_orig(M2ROI_FN):
    def __init__(self, version='v1'):
        super(M2ROI_FN_orig, self).__init__(version)
        # self.share = self.dict_train[version]['share']
        for m in self.modality_list:
            for roi in self.rois_list:
                roi = str(roi)
                self.cnn_dict[m]['conv_' + roi] = BaseModel().conv

    def forward(self, x):
        features_dict = {}
        for m in self.modality_list:
            features_dict[m] = {}
            for roi in self.rois_list:
                roi = str(roi)
                low_level_local_features = self.cnn_dict[m]['conv_{}'.format(roi)](x[roi][m].float().to(self.device))
                low_level_local_features = nn.Flatten()(self.cnn_dict[m][roi](low_level_local_features))
                features_dict[m][roi] = low_level_local_features

        features_dict['high_level'] = {}
        if len(self.modality_list) > 1:
            for roi in self.rois_list:
                roi = str(roi)
                temp_features = torch.cat(
                    [features_dict[self.modality_list[0]][roi],
                     features_dict[self.modality_list[1]][roi],
                     ], dim=1
                )
                features_dict['high_level'][roi] = self.mni[roi](temp_features)
        else:
            features_dict['high_level'] = features_dict[self.modality_list[0]]
        if self.morphological:
            features_dict['morphological'] = []
            for roi in self.rois_list:
                roi = str(roi)
                features_dict['morphological'].append(x[roi]['mor'])
            features_dict['morphological'] = torch.cat(features_dict['morphological'], dim=1)
        else:
            features_dict['morphological'] = torch.tensor([])
        global_features = torch.cat([features_dict['high_level'][i] for i in features_dict['high_level'].keys()] + [
            features_dict['morphological'].float().to(self.device)], dim=1)
        return self.ROI_based_MLP_classifier(global_features)


if __name__ == '__main__':
    # input_ = {
    #     'lh-middletemporal':
    #         {
    #             'T1': torch.randn(2, 1, 44, 68, 113),
    #             'pet': torch.randn(2, 1, 44, 68, 113),
    #             'mor': torch.randn(2, 4)
    #
    #         },
    #     # 'Right-Hippocampus':
    #     #     {
    #     #         'T1': torch.randn(2, 1, 34, 36, 47),
    #     #         'pet': torch.randn(2, 1, 34, 36, 47),
    #     #         'mor': torch.randn(2, 1)
    #     #     }
    # }
    input_ = torch.randn(2, 1, 33, 37, 51)
    net = BaseModel()
    with torch.no_grad():
        out = net.forward(input_)
    print(out.shape)
