import os
import numpy as np
import pandas as pd
from utils import dict_label, dict_max_shape, dict_train, get_columns, dict_data_table_columns
# from utils import dict_train_2 as dict_train
import nibabel as nib
from torch.utils.data import Dataset
import random
import json


class get_patch(object):
    def __init__(self, roi, save_path):
        frame = pd.read_csv(r'My_adni_table3_1.csv')
        frame = frame[~pd.isna(frame['Pet_path'])]
        self.frame = frame[pd.isna(frame['M2ROI_FN_path'])]
        self.index = self.frame.index  # [:3]
        self.save_path = save_path
        if roi in list(dict_label.keys()):
            self.roi_information = dict_label[roi]
        else:
            import sys
            print('{} not in utils.py/dict_label!'.format(roi))
            sys.exit()

    @staticmethod
    def fun_get_patch(image, label, num):
        mask = np.zeros_like(label)
        mask[label == num] = 1
        # image_mask_3D = image_3D * mask_3D
        indices = np.argwhere(mask == 1)
        # center = np.mean(indices, axis=0).astype(int)
        x_max = np.max(indices[:, 0]).astype(int)
        x_min = np.min(indices[:, 0]).astype(int)
        y_max = np.max(indices[:, 1]).astype(int)
        y_min = np.min(indices[:, 1]).astype(int)
        z_max = np.max(indices[:, 2]).astype(int)
        z_min = np.min(indices[:, 2]).astype(int)

        image_crop_3D = image[x_min:x_max, y_min:y_max, z_min:z_max]
        mask_crop_3D = mask[x_min:x_max, y_min:y_max, z_min:z_max]

        # import matplotlib.pyplot as plt
        # plt.imshow(image_crop_3D[10, ...])
        # plt.show()
        return image_crop_3D, mask_crop_3D

    @staticmethod
    def compute_value(image, label, num):
        mask = np.zeros_like(label)
        mask[label == num] = 1
        return np.round(np.sum(image * mask), 3)

    @staticmethod
    def compute_mean(surf_mor, surf_label, num):
        mask = np.zeros_like(surf_label)
        mask[surf_label == num] = 1
        return np.round(np.sum(surf_mor * mask) / np.sum(mask), 3)

    @staticmethod
    def compute_standard_deviation(surf_mor, surf_label, num):
        mask = np.zeros_like(surf_label)
        mask[surf_label == num] = 1
        data = surf_mor * mask
        data_mean = np.sum(data) / np.sum(mask)
        squared_diffs = [(x - data_mean) ** 2 for x in data if x != 0]
        variance = np.sum(squared_diffs) / np.sum(mask)
        std_dev = variance ** 0.5
        return np.round(std_dev, 3)

    def __len__(self):
        return len(self.index.tolist())
        # return 3

    def __getitem__(self, item):
        subject_id = self.frame.loc[self.index[item], 'Subject ID']
        path_T1 = self.frame.loc[self.index[item], 'FreeSurfer_path']
        path_pet = self.frame.loc[self.index[item], 'Pet_path']
        if os.path.isdir(os.path.join(path_T1, 'surf')):
            T1_image = nib.load(os.path.join(path_T1, 'mri', 'norm.mgz')).get_fdata()
            pet_image = nib.load(os.path.join(path_pet, 'PET_to_T1.nii.gz')).get_fdata()
            label = nib.load(os.path.join(path_T1, 'mri', 'aparc+aseg.mgz')).get_fdata()
            atlas_name, roi_name, orientation, roi, start_point = self.roi_information
            T1_patch, mask_patch = self.fun_get_patch(T1_image, label, roi + start_point)
            pet_patch, _ = self.fun_get_patch(pet_image, label, roi + start_point)
            mor = pd.DataFrame()
            if atlas_name == 'aseg':
                vim = self.compute_mean(T1_image, label, roi)
                mor.insert(0, 'vim', np.nan)
                mor.loc[0, 'vim'] = vim

            elif atlas_name == 'aparc':
                surf_label = nib.freesurfer.read_annot(
                    os.path.join(path_T1, 'label', '{}.aparc.annot').format(orientation))
                surf_thickness = nib.freesurfer.read_morph_data(
                    os.path.join(path_T1, 'surf', '{}.thickness'.format(orientation)))
                surf_volume = nib.freesurfer.read_morph_data(
                    os.path.join(path_T1, 'surf', '{}.volume'.format(orientation)))
                surf_area = nib.freesurfer.read_morph_data(os.path.join(path_T1, 'surf', '{}.area'.format(orientation)))
                # surf_curv = nib.freesurfer.read_morph_data(os.path.join(path_T1, 'surf', '{}.curv'.format(orientation)))
                # surf_sulc = nib.freesurfer.read_morph_data(os.path.join(path_T1, 'surf', '{}.sulc'.format(orientation)))
                cta = self.compute_mean(surf_thickness, surf_label[0], roi)
                ctd = self.compute_standard_deviation(surf_thickness, surf_label[0], roi)
                gmv = self.compute_value(surf_volume, surf_label[0], roi)
                sa = self.compute_value(surf_area, surf_label[0], roi)
                # curv = self.compute_value(surf_curv, surf_label[0], roi)
                # sulc = self.compute_value(surf_sulc, surf_label[0], roi)
                for name, val_ in zip(['cta', 'ctd', 'gmv', 'sa'], [cta, ctd, gmv, sa]):
                    mor.insert(0, name, np.nan)
                    mor.loc[0, name] = val_

            if not os.path.exists(os.path.join(self.save_path, subject_id)):
                os.makedirs(os.path.join(self.save_path, subject_id))
            np.save(os.path.join(self.save_path, subject_id, '{}_T1.npy'.format(roi_name)), T1_patch)
            np.save(os.path.join(self.save_path, subject_id, '{}_pet.npy'.format(roi_name)), pet_patch)
            np.save(os.path.join(self.save_path, subject_id, '{}_mask.npy'.format(roi_name)), mask_patch)
            mor.to_csv(os.path.join(self.save_path, subject_id, '{}_mor.csv'.format(roi_name)), index=False)

            print('{} finish!'.format(subject_id))
        else:
            print('{} recon-all error! surf not found!'.format(subject_id))
            pass


class dataProc(Dataset):
    def __init__(self, stage='train', fold=0, seed=2023, group='sMCI_pMCI', frame_path=r'T1_FGD.csv',
                 version='v1', normalization=True, data_path='/media/shucheng/MyBook/DL_dataset/M2ROI-FN'):
        self.stage = stage
        self.mask = dict_train[version]['mask']
        self.normalization = normalization
        frame = pd.read_csv(frame_path)
        self.frame = frame.copy()
        frame = frame[~pd.isna(frame['Data path.T1']) & ~pd.isna(frame['Data path.pet.fgd']) & frame['Group'].isin(
            group.split('_'))]
        index_ = frame.index.tolist()
        random.seed(seed)
        random.shuffle(index_)
        if stage == 'train':
            temp_ = [np.array_split(frame.index.tolist(), 5)[i] for i in range(5) if i != fold]
            self.index = np.concatenate(temp_, axis=0)
        else:
            self.index = np.array_split(frame.index.tolist(), 5)[fold]
        self.dict = {
            group.split('_')[0]: 0,
            group.split('_')[1]: 1,
        }
        self.rois = dict_train[version]['rois']
        self.mean_std = self.load_json('mean_std.json')
        self.max_shape = self.load_json('max_shape.json')

        self.data_path = data_path
        self.morphological_table = self.load_mor_table()
        self.mor_column_name = dict_data_table_columns

    def load_mor_table(self):
        frame = pd.read_csv(r'data_table.csv', usecols=get_columns())
        frame.fillna(frame.mean(), inplace=True)
        return (frame - frame.mean()) / frame.std()

    def load_json(self, json_path):
        with open(json_path, 'r') as file:
            return json.load(file)

    def norm_image(self, image, modality, roi):
        return (image - np.mean(image)) / np.std(image)
        # return (image - self.mean_std[modality][roi][0]) / self.mean_std[modality][roi][1]

    def padding(self, patch, roi):
        x, y, z = patch.shape
        max_shape = self.max_shape[roi]
        max_shape_patch = np.zeros(max_shape)
        max_shape_patch[:x, :y, :z] = patch
        return max_shape_patch

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, item):
        subject_information = self.frame.loc[self.index[item], :]
        morphological_information = self.morphological_table.loc[self.index[item], :]
        path = os.path.join(self.data_path, subject_information['Subject ID'])
        data_dict = {}
        for roi in self.rois:
            roi = str(roi)
            T1_patch = np.load(os.path.join(path, f'{roi}_T1.npy'))
            pet_patch = np.load(os.path.join(path, f'{roi}_Pet.npy'))
            mor_patch = morphological_information[self.mor_column_name[roi]].values
            if self.normalization:
                T1_patch = self.norm_image(T1_patch, 'T1', roi)
                pet_patch = self.norm_image(pet_patch, 'Pet', roi)

            if self.mask:
                mask_patch = np.load(os.path.join(path, f'{roi}_mask.npy'))
                T1_patch = mask_patch * T1_patch
                pet_patch = mask_patch * pet_patch

            max_shape_pet_patch = self.padding(pet_patch, roi)
            max_shape_T1_patch = self.padding(T1_patch, roi)

            data_dict[roi] = {
                'pet': max_shape_pet_patch[None, ...],
                'T1': max_shape_T1_patch[None, ...],
                'mor': mor_patch,
            }
        label = self.dict[subject_information['Group']]
        return data_dict, label

    def to_label(self, group_list):
        label = []
        for i in group_list:
            cls = self.dict[i]
            label.append(cls)
        return label

    def get_sample_weights(self):
        weights = []
        count_nums = np.arange(0, 2).astype(np.int64)
        count = float(self.index.shape[0])
        label = self.to_label(self.frame.loc[self.index.tolist(), 'Group'])
        count_class_list = [float(label.count(i)) for i in count_nums]
        for i in label:
            for j in count_nums:
                if i == j:
                    weights.append(count / count_class_list[j])
        imbalanced_ratio = [count_class_list[0] / i_r for i_r in count_class_list]
        return weights, imbalanced_ratio


if __name__ == '__main__':
    # for roi_ in dict_label.keys():
    #     print('*' * 30 + roi_ + '*' * 30)
    #     dp = get_patch(roi=roi_, save_path='/media/czp/MyBook/dataset/M2ROI_FN_data')
    #     for i in dp:
    #         pass

    dp = dataProc(group='sMCI_pMCI')
    a, b = dp.get_sample_weights()
    for i in dp:
        pass
