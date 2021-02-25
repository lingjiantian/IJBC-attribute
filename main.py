import pandas as pd
import os
import numpy as np

def init_from_list(filename, folder_depth=2):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    abspaths = [os.path.abspath(line[0]) for line in lines]
    paths = ['/'.join(p.split('/')[-folder_depth:]) for p in abspaths]
    if len(lines[0]) == 2:
        labels = [int(line[1]) for line in lines]
        names = [str(lb) for lb in labels]
    elif len(lines[0]) == 1:
        names = [p.split('/')[-folder_depth] for p in abspaths]
        _, labels = np.unique(names, return_inverse=True)
    else:
        raise ValueError('List file must be in format: "fullpath(str) \
                                    label(int)" or just "fullpath(str)"')

    data = pd.DataFrame({'path': paths, 'abspath': abspaths,
                              'label': labels, 'name': names})
    prefix = abspaths[0].split('/')[:-folder_depth]
    return data, prefix

init_from_list("data/fileList.txt")