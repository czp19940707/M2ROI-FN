import pandas as pd

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


if __name__ == '__main__':
    # frame = pd.read_csv(r'data_table.csv')
    # dict_mor_mean_std = {}
    # for roi in dict_data_table_columns.keys():
    #     for column_name in dict_data_table_columns[roi]:
    #         mean_ = frame[roi].mean()
    aa = get_columns()
    print()
