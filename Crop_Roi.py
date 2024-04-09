import nibabel as nib
import pandas as pd
import os
import argparse
import numpy as np

# roi_ctx = [1006, 2006, 1015, 2015, 1035, 2035]
# roi_subctx = [17, 18, 53, 54]

roi_ctx = [1032, 2032]
roi_subctx = []

rois = roi_ctx + roi_subctx


def fun_get_patch(label, num):
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
    return x_max, x_min, y_max, y_min, z_max, z_min


def vis(data1, data2, slice_num=13):
    # 创建一个图和两个子图
    import matplotlib.pyplot as plt
    data1 = data1[slice_num, ...]
    data2 = data2[slice_num, ...]
    plt.figure(figsize=(10, 5))  # 设置图的大小，宽10英寸，高5英寸

    # 显示第一个数组
    plt.subplot(1, 2, 1)  # 1行2列的第1个子图
    plt.imshow(data1, cmap='viridis')
    plt.colorbar()
    plt.title('Array 1')

    # 显示第二个数组
    plt.subplot(1, 2, 2)  # 1行2列的第2个子图
    plt.imshow(data2, cmap='viridis')
    plt.colorbar()
    plt.title('Array 2')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='/media/shucheng/MyBook/2024_ADNI_part3/127_S_4198', type=str)
    args = parser.parse_args()

    # frame = pd.read_csv(r'T1_FGD.csv')
    # frame = frame[~pd.isna(frame['Data path.T1']) & ~pd.isna(frame['Data path.pet.fgd'])]
    # for index in frame.index:
    # subject_information = frame.loc[index, :]
    subject_id = os.path.split(args.i)[-1]
    save_path = f'/media/shucheng/MyBook/DL_dataset/M2ROI-FN/{subject_id}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_path_pet_fgd = os.path.join(args.i, 'pet', 'fgd_Corg', 'template2rainmask_dof12.nii.gz')
    # data_path_pet_fgd_subctx = os.path.join(args.i, 'pet', 'fgd_pvc', 'gtmpvc.output',
    #                                         'mgx.subctxgm.mni305.1mm.sm00.nii.gz')
    data_path_label = os.path.join(args.i, 'fs', 'T1', 'mri', 'aparc+aseg.mgz')
    data_path_T1 = os.path.join(args.i, 'fs', 'T1', 'mri', 'norm.mgz')
    label = nib.load(data_path_label).get_fdata()
    T1 = nib.load(data_path_T1).get_fdata()
    pet_fdg = nib.load(data_path_pet_fgd).get_fdata()
    # vis(T1, pet_fdg, slice_num=130)

    # pet_fdg_subctx = nib.load(data_path_pet_fgd_subctx).get_fdata()

    for roi in rois:
        x_max, x_min, y_max, y_min, z_max, z_min = fun_get_patch(label, roi)
        T1_crop_3D = T1[x_min:x_max, y_min:y_max, z_min:z_max]
        PET_Crop_3D = pet_fdg[x_min:x_max, y_min:y_max, z_min:z_max]
        mask_crop_3D = label[x_min:x_max, y_min:y_max, z_min:z_max]
        # vis(T1_crop_3D, PET_Crop_3D)
        np.save(os.path.join(save_path, f'{roi}_T1.npy'), T1_crop_3D)
        np.save(os.path.join(save_path, f'{roi}_Pet.npy'), PET_Crop_3D)
        np.save(os.path.join(save_path, f'{roi}_Mask.npy'), mask_crop_3D)

    # for roi in roi_ctx:
    #     x_max, x_min, y_max, y_min, z_max, z_min = fun_get_patch(label, roi)
    #     T1_crop_3D = T1[x_min:x_max, y_min:y_max, z_min:z_max]
    #     PET_ctx_Crop_3D = pet_fdg_ctx[x_min:x_max, y_min:y_max, z_min:z_max]
    #     mask_crop_3D = label[x_min:x_max, y_min:y_max, z_min:z_max]
    #     np.save(os.path.join(save_path, f'{roi}_T1.npy'), T1_crop_3D)
    #     np.save(os.path.join(save_path, f'{roi}_Pet.npy'), PET_ctx_Crop_3D)
    #     np.save(os.path.join(save_path, f'{roi}_Mask.npy'), mask_crop_3D)
