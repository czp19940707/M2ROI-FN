import pandas as pd

if __name__ == '__main__':
    # path2 = r'/home/shucheng/python_files/Work/PET-FDG/idaSearch_2_26_2024.csv'
    # path1 = r'/home/shucheng/python_files/DL_classifier/M2ROI-FN/T1_FGD.csv'
    # df1 = pd.read_csv(path1)
    # df2 = pd.read_csv(path2)
    # df_merged = pd.merge(df1, df2, left_on='Image ID.pet.fdg', right_on='Image ID', how='left')
    # df_merged.to_csv(r'T1_FGD_demographic.csv', index=False)
    frame_copy = pd.read_csv(r'T1_FGD_demographic.csv').copy()
    demographic_stats_tabel = pd.DataFrame(
        columns=['Group', 'Num', 'Age_mean', 'Age_std', 'Age_max', 'Age_min', 'Gender_M', 'Gender_F', 'MMSE_mean',
                 'MMSE_std', 'MMSE_max', 'MMSE_min'])
    frame_copy = frame_copy[frame_copy['Group.pet.fdg'].isin(['sMCI', 'CN', 'AD', 'pMCI'])]
    for count_, group in enumerate(['CN', 'AD', 'sMCI', 'pMCI']):
        frame = frame_copy[frame_copy['Group.pet.fdg'] == group]

        demographic_stats_tabel.loc[count_, 'Group'] = group
        demographic_stats_tabel.loc[count_, 'Num'] = len(frame)
        demographic_stats_tabel.loc[count_, 'Gender_M'] = len(frame[frame['Sex'] == 'M'])
        demographic_stats_tabel.loc[count_, 'Gender_F'] = len(frame[frame['Sex'] == 'F'])
        demographic_stats_tabel.loc[count_, 'Age_mean'] = frame['Age'].mean()
        demographic_stats_tabel.loc[count_, 'Age_std'] = frame['Age'].std()
        demographic_stats_tabel.loc[count_, 'Age_max'] = frame['Age'].max()
        demographic_stats_tabel.loc[count_, 'Age_min'] = frame['Age'].min()
        demographic_stats_tabel.loc[count_, 'MMSE_mean'] = frame['MMSE Total Score'].mean()
        demographic_stats_tabel.loc[count_, 'MMSE_std'] = frame['MMSE Total Score'].std()
        demographic_stats_tabel.loc[count_, 'MMSE_max'] = frame['MMSE Total Score'].max()
        demographic_stats_tabel.loc[count_, 'MMSE_min'] = frame['MMSE Total Score'].min()

    demographic_stats_tabel.loc[count_ + 1, 'Group'] = 'Total'
    demographic_stats_tabel.loc[count_ + 1, 'Num'] = len(frame_copy)
    demographic_stats_tabel.loc[count_ + 1, 'Gender_M'] = len(frame_copy[frame_copy['Sex'] == 'M'])
    demographic_stats_tabel.loc[count_ + 1, 'Gender_F'] = len(frame_copy[frame_copy['Sex'] == 'F'])
    demographic_stats_tabel.loc[count_ + 1, 'Age_mean'] = frame_copy['Age'].mean()
    demographic_stats_tabel.loc[count_ + 1, 'Age_std'] = frame_copy['Age'].std()
    demographic_stats_tabel.loc[count_ + 1, 'Age_max'] = frame_copy['Age'].max()
    demographic_stats_tabel.loc[count_ + 1, 'Age_min'] = frame_copy['Age'].min()
    demographic_stats_tabel.loc[count_ + 1, 'MMSE_mean'] = frame_copy['MMSE Total Score'].mean()
    demographic_stats_tabel.loc[count_ + 1, 'MMSE_std'] = frame_copy['MMSE Total Score'].std()
    demographic_stats_tabel.loc[count_ + 1, 'MMSE_max'] = frame_copy['MMSE Total Score'].max()
    demographic_stats_tabel.loc[count_ + 1, 'MMSE_min'] = frame_copy['MMSE Total Score'].min()
    demographic_stats_tabel.to_csv('demographic_stats_tabel.csv', index=False)
