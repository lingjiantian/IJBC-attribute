"""Test PFE on IJB-A.
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

import os
import sys
import time
import math
import argparse
import numpy as np

from utils import utils
from utils.dataset import Dataset
from utils.imageprocessing import preprocess
from tqdm import tqdm

from ijba import IJBATest
from ijbc import IJBCTest
import pdb

def aggregate_templates(templates, features, method):
    for i,t in tqdm(enumerate(templates)):
        if len(t.indices) > 0:
            if method == 'mean':
                t.feature = utils.l2_normalize(np.mean(features[t.indices], axis=0))
            if method == 'sample':
                t.feature = utils.random_sample(features[t.indices], normalized=True)
        else:
            t.feature = None
        # if i % 1000 == 0:
        #     sys.stdout.write('Fusing templates {}%...\t\r'.format(i/len(templates)*100))
    print('')


def force_compare(compare_func, verbose=False):
    def compare(t1, t2):
        score_vec = np.zeros(len(t1))
        for i in range(len(t1)):
            if t1[i] is None or t2[i] is None:
                score_vec[i] = -9999
            else:
                score_vec[i] = compare_func(t1[i][None], t2[i][None])
            if verbose and i % 1000 == 0:
                sys.stdout.write('Matching pair {}/{}...\t\r'.format(i, len(t1)))
        if verbose:
            print('')
        return score_vec
    return compare

def readDat(feat_path, byte_per_float = 4):
    """
    读取特征文件
    :param feat_path: 特征文件路径
    :return:
    """
    feat = open(feat_path, 'rb')

    info_file = np.fromfile(feat, dtype=np.int32, count=2)
    dim = info_file[0]
    image_num = info_file[1]
    info_length = byte_per_float * 2

    feat.seek(info_length)
    feat_data = np.fromfile(feat, dtype=np.float32, count=image_num * dim)

    features = []
    for i in range(image_num):
        features.append(np.array(feat_data[i*dim:(i+1)*dim]))
    return np.array(features)


def main(args):
    recoder = open(args.recode_file, "a")
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, file=recoder)

    if args.protocol == 'ijba':
        tester = IJBATest(args.file_list)
        tester.init_proto(args.protocol_path)
    elif args.protocol == 'ijbc':
        tester = IJBCTest(args.file_list, index=3, thre=0.2)  # 低画质
        # tester = IJBCTest(args.file_list, index=-1, thre=40)    # 大姿态
        # tester = IJBCTest(args.file_list, index=None)
        tester.init_proto(args.protocol_path)
    else:
        raise ValueError('Unkown protocol. Only accept "ijba" or "ijbc".')

    if os.path.isdir(args.dat_dir):
        # files = [i for i in os.listdir(args.dat_dir) if "feature" in i and os.path.splitext(i)[1] == ".dat"]
        files = [i for i in os.listdir(args.dat_dir) if os.path.splitext(i)[1] == ".dat"]
        files = [os.path.join(args.dat_dir, i) for i in files]
    else:
        files = [args.dat_dir]
    print(files)

    for file in files:
        print("dataset:%s\nmodel:%s" % (args.protocol, file), file=recoder)
        features = readDat(file)

        print('---- Average pooling', file=recoder)
        aggregate_templates(tester.verification_templates, features, 'mean')
        TARs, std, FARs = tester.test_verification(force_compare(utils.pair_euc_score))
        for i in range(len(TARs)):
            print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]), file=recoder)

        # print('---- Random sampling')
        # aggregate_templates(tester.verification_templates, features, 'sample')
        # TARs, std, FARs = tester.test_verification(force_compare(utils.pair_euc_score))
        # for i in range(len(TARs)):
        #     print('TAR: {:.5} +- {:.5} FAR: {:.5}'.format(TARs[i], std[i], FARs[i]))
        recoder.flush()
    recoder.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", help="The dataset to test",
                        type=str, default='ijbc')
    parser.add_argument("--file_list", help="The path to the IJB-A dataset directory",
                        # type=str, default='data/ijba_fileList.txt')
                        type=str, default='data/ijbc_pose.txt')
    parser.add_argument("--protocol_path", help="The path to the IJB-A protocol directory",
                        type=str, default='data/metaData/ijbc')
    parser.add_argument("--dat_dir", help="the feature dat file",
                        type=str, default='data/dat/ijcai2021/')

    parser.add_argument("--recode_file",
                        type=str, default='./recode.txt')
    args = parser.parse_args()
    main(args)
