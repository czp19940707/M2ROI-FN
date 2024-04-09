import pandas as pd
import os

if __name__ == '__main__':
    frame = pd.read_csv(r'/Work/PET-FDG/T1_FDG1.csv')
    count = 0
    for index_ in frame.index:
        subject_information = frame.loc[index_, :]
        data_path = subject_information['Data path']
        if os.path.isfile(os.path.join(data_path, 'fs', 'T1', 'surf', 'lh.thickness')):
            frame.loc[index_, 'Data path.T1'] = 1

    frame.to_csv(r'T1_FGD.csv', index=False)