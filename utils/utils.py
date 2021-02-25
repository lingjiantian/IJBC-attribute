"""Utilities for training and testing
"""
# MIT License
# 
# Copyright (c) 2019 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import numpy as np
from scipy import misc
import time
import math
import random
from datetime import datetime
import shutil
import pandas as pd
from collections import defaultdict

def create_log_dir(config, config_file):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(config.log_base_dir), config.name, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    shutil.copyfile(config_file, os.path.join(log_dir,'config.py'))

    return log_dir


def get_updated_learning_rate(global_step, config):
    if config.learning_rate_strategy == 'step':
        max_step = -1
        learning_rate = 0.0
        for step, lr in config.learning_rate_schedule.items():
            if global_step >= step and step > max_step:
                learning_rate = lr
                max_step = step
        if max_step == -1:
            raise ValueError('cannot find learning rate for step %d' % global_step)
    elif config.learning_rate_strategy == 'cosine':
        initial = config.learning_rate_schedule['initial']
        interval = config.learning_rate_schedule['interval']
        end_step = config.learning_rate_schedule['end_step']
        step = math.floor(float(global_step) / interval) * interval
        assert step <= end_step
        learning_rate = initial * 0.5 * (math.cos(math.pi * step / end_step) + 1)
    elif config.learning_rate_strategy == 'linear':
        initial = config.learning_rate_schedule['initial']
        start = config.learning_rate_schedule['start']
        end_step = config.learning_rate_schedule['end_step']
        assert global_step <= end_step
        assert start < end_step
        if global_step < start:
            learning_rate = initial
        else:
            learning_rate = 1.0 * initial * (end_step - global_step) / (end_step - start)
    else:
        raise ValueError("Unkown learning rate strategy!")

    return learning_rate

def display_info(epoch, step, duration, watch_list):
    sys.stdout.write('[%d][%d] time: %2.2f' % (epoch+1, step+1, duration))
    for item in watch_list.items():
        if type(item[1]) in [float, np.float32, np.float64]:
            sys.stdout.write('   %s: %2.3f' % (item[0], item[1]))
        elif type(item[1]) in [int, bool, np.int32, np.int64, np.bool]:
            sys.stdout.write('   %s: %d' % (item[0], item[1]))
    sys.stdout.write('\n')

def l2_normalize(x, axis=None, eps=1e-8):
    x = x / (eps + np.linalg.norm(x, axis=axis))
    return x

def random_sample(x, normalized=True):
    sampleId = random.randint(0, len(x)-1)
    x = x[sampleId]
    if normalized:
        x = l2_normalize(x)
    return x

def pair_euc_score(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist

def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert sigma_sq2 is None, 'either pass in concated features, or mu, sigma_sq for both!'
        D = int(x1.shape[1] / 2)
        mu1, sigma_sq1 = x1[:,:D], x1[:,D:]
        mu2, sigma_sq2 = x2[:,:D], x2[:,D:]
    else:
        x1, x2 = np.array(x1), np.array(x2)
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)
    return -dist


def aggregate_PFE(x, sigma_sq=None, normalize=True, concatenate=False):
    if sigma_sq is None:
        D = int(x.shape[1] / 2)
        mu, sigma_sq = x[:,:D], x[:,D:]
    else:
        mu = x
    attention = 1. / sigma_sq
    attention = attention / np.sum(attention, axis=0, keepdims=True)
    
    mu_new  = np.sum(mu * attention, axis=0)
    sigma_sq_new = np.min(sigma_sq, axis=0)

    if normalize:
        mu_new = l2_normalize(mu_new)

    if concatenate:
        return np.concatenate([mu_new, sigma_sq_new])
    else:
        return mu_new, sigma_sq_new
    

def create_ijbc_file_list(meta_file_path, save_path, feature_save_path):
    file = pd.read_csv(meta_file_path)
    fileName_subject = defaultdict(int)
    with open(save_path, "w") as f:
        with open(feature_save_path, "w") as ff:
            for index, row in file.iterrows():
                    if (not np.isnan(row["SUBJECT_ID"])) and ("video" not in row["FILENAME"]):
                        if row["FILENAME"] not in fileName_subject:
                            fileName_subject[row["FILENAME"]] = 0
                        else:
                            fileName_subject[row["FILENAME"]]+=1

                        s1, s2 = os.path.splitext(row["FILENAME"])
                        newFileName = s1 + "%d" % fileName_subject[row["FILENAME"]] + s2
                        f.write("%s %.0f %.0f %.0f %.0f %d %s\n"%(row["FILENAME"], row["FACE_X"], row["FACE_Y"],
                                                     row["FACE_WIDTH"], row["FACE_HEIGHT"], row["SUBJECT_ID"], newFileName))

                        ff.write("%s 79 99 79 99 %d\n" % (newFileName, row["SUBJECT_ID"]))
                    # if np.isnan(row["FACE_X"]) or \
                    #         np.isnan(row["FACE_WIDTH"]) or np.isnan(row["FACE_HEIGHT"]) and not np.isnan(row["FACE_Y"]):
                    #     print("%s\n"%(fileName))


def create_ijba_file_list(meta_file_root, save_path, feature_save_path):
    temp = set()
    fileName_subject = defaultdict(int)
    with open(save_path, "w") as f:
        with open(feature_save_path, "w") as ff:
            for i in range(1,11):
                print("deal %d split..."%i)
                metadata_path = os.path.join(meta_file_root, "split{split}/verify_metadata_{split}.csv".format(split=i))
                train_path = os.path.join(meta_file_root, "split{split}/train_{split}.csv".format(split=i))
                for file in [metadata_path, train_path]:
                    fd = pd.read_csv(file)
                    for index, row in fd.iterrows():
                        if not np.isnan(row["SUBJECT_ID"]) and not np.isnan(row["FACE_X"]) and \
                                not np.isnan(row["FACE_WIDTH"]) and not np.isnan(row["FACE_HEIGHT"]) and not np.isnan(row["FACE_Y"]):

                            info = "%s %d %d %d %d %d\n" % (row["FILE"], row["FACE_X"], row["FACE_Y"],
                                                             row["FACE_WIDTH"], row["FACE_HEIGHT"], row["SUBJECT_ID"])
                            if info not in temp:
                                temp.add(info)

                                if row["FILE"] not in fileName_subject:
                                    fileName_subject[row["FILE"]] = 0
                                else:
                                    fileName_subject[row["FILE"]] += 1

                                s1, s2 = os.path.splitext(row["FILE"])
                                newFileName = s1 + "%d" % fileName_subject[row["FILE"]] + s2
                                f.write("%s %d %d %d %d %d %s\n" % (row["FILE"], row["FACE_X"], row["FACE_Y"],
                                                             row["FACE_WIDTH"], row["FACE_HEIGHT"], row["SUBJECT_ID"],newFileName))



                                ff.write("%s 79 99 79 99 %d\n" % (newFileName, row["SUBJECT_ID"]))

if __name__ == '__main__':
    # ijbc
    meta_file_path = "../data/metaData/ijbc/ijbc_metadata.csv"
    save_path = "../data/ijbc_fileList.txt"
    feature_save_path = "../data/ijbc_feature_fileList.txt"
    create_ijbc_file_list(meta_file_path, save_path, feature_save_path)

    # ijba
    meta_file_root = "../data/metaData/ijba/IJB-A_11_sets"
    save_path = "../data/ijba_fileList.txt"
    feature_save_path = "../data/ijba_feature_fileList.txt"
    create_ijba_file_list(meta_file_root, save_path, feature_save_path)