import subprocess
import pandas as pd

if __name__ == '__main__':
    commands = []
    frame = pd.read_csv(r'T1_FGD.csv')
    frame = frame[~pd.isna(frame['Data path.T1']) & ~pd.isna(frame['Data path.pet.fgd'])]
    for index_ in frame.index:
        Data_path = frame.loc[index_, 'Data path']
        command = f'python Crop_Roi.py -i {Data_path}'
        commands.append(command)

    process = subprocess.Popen(['parallel', '-j', '32', '--gnu', ':::'] + commands)
    process.wait()