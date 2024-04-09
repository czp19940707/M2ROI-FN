import torch

from model import BaseModel
import json

if __name__ == '__main__':
    with open('max_shape.json', 'r') as f:
        dict_max_shape = json.load(f)

    net = BaseModel()
    dict_save = {}
    for Roi in dict_max_shape.keys():
        input_ = torch.randn(dict_max_shape[Roi])[None, None, ...]
        out = net.forward(input_)
        dict_save[Roi] = out.shape[2:]

    with open('conv_kernal_shape.json', 'w') as f1:
        json.dump(dict_save, f1, indent=4)