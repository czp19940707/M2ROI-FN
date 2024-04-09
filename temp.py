import os
import pandas as pd

if __name__ == '__main__':
    morphological_ctx = ['Meanpial_lgi', 'ThickAvg', 'SurfArea', 'GausCurv', 'GrayVol']
    morphological_subctx = ['Volume_mm3', 'Mean']
    atlas_ctx = ['aparc']
    atlas_subctx = ['aseg']
    roi_ctx = ['entorhinal', 'insula', 'middletemporal', 'frontalpole']
    roi_subctx = ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala']
    path = r'/home/shucheng/python_files/ML_classifier/data'

    frame_base = pd.read_csv(r'T1_FGD.csv')

    for side in ['lh', 'rh']:
        for morph_ctx in morphological_ctx:
            for at_ctx in atlas_ctx:
                frame = pd.read_csv(os.path.join(path, f'{side}.{morph_ctx}.{at_ctx}.csv'))
                # for index_ in frame.index:
                #     subject_id = frame.loc[index_, 'Subject ID']
                for roi in roi_ctx:
                    column_name = f'{side}.{morph_ctx}.{at_ctx}.{roi}'
                    frame_base[column_name] = frame[roi]


    for morph_subctx in morphological_subctx:
        for at_subctx in atlas_subctx:
            frame = pd.read_csv(os.path.join(path, f'{morph_subctx}.{at_subctx}.csv'))
            for roi in roi_subctx:
                column_name = f'{morph_subctx}.{at_subctx}.{roi}'
                frame_base[column_name] = frame[roi]

    frame_base.to_csv(r'data_table.csv', index=False)