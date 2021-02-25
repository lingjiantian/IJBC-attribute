"""
Main file for evaluation on IJB-A and IJB-B protocols.
More instructions can be found in README.md file.
2017 Yichun Shi
"""

import sys
import os
import numpy as np
import utils

import metrics
from collections import namedtuple


# Configuration
VerificationFold = namedtuple('VerificationFold', ['train_indices', 'test_indices', 'train_templates', 'templates1','templates2'])

class Template:
    def __init__(self, template_id, label, indices, medias):
        self.template_id = template_id
        self.label = label
        self.indices = np.array(indices)
        self.medias = np.array(medias)
        

def build_subject_dict(file_list, index=None):
    subject_dict = {}
    with open(file_list, "r") as f:
        for i, line in enumerate(f.readlines()[1:]):
            line = line.strip().split(" ")
            subject_id, image = int(line[1]), line[0]
            image = image.split(".")
            image = image[0][:-1]+"."+image[1]
            if not subject_id in subject_dict:
                subject_dict[subject_id] = {}
            if index is not None:
                subject_dict[subject_id][image] = [i, float(line[index])]
            else:
                subject_dict[subject_id][image] = [i, None]

    return subject_dict

def build_templates(subject_dict, meta_file, gallery = True, thre = None):
    with open(meta_file, 'r') as f:
        meta_list = f.readlines()
        meta_list = [x.split('\n')[0] for x in meta_list]
        meta_list = meta_list[1:]

    templates = []
    template_id = None
    template_label = None
    template_indices = None
    template_medias = None
    count = 0
    miss_count = 0
    for line in meta_list:
        temp_id, subject_id, image, media = tuple(line.split(',')[0:4])
        temp_id = int(temp_id)
        subject_id = int(subject_id)
        if subject_id in subject_dict and image in subject_dict[subject_id]:
            values = subject_dict[subject_id][image]
            if values[1] is None:   # 无需比较
                index = subject_dict[subject_id][image][0]
                # count += 1
            elif gallery:
                if values[1] > thre:    # 低画质
                # if abs(values[1]) < thre:  # 大姿态
                    index = subject_dict[subject_id][image][0]
                    # count += 1
                else:
                    index = None
            elif not gallery:
                if values[1] < thre:  # 低画质
                # if abs(values[1]) > thre:    # 大姿态
                    index = subject_dict[subject_id][image][0]
                    # count += 1
                else:
                    index = None

        else:
            index = None

        if temp_id != template_id:
            if template_id is not None:
                if len(template_indices) == 0:
                    miss_count += 1
                    # templates.append(Template(template_id, template_label, template_indices, template_medias))
                else:
                    count += 1
                    templates.append(Template(template_id, template_label, template_indices, template_medias))

            template_id = temp_id
            template_label = subject_id
            template_indices = []
            template_medias = []

        if index is not None:
            template_indices.append(index)        
            template_medias.append(media)        

    # last template
    templates.append(Template(template_id, template_label, template_indices, template_medias))
    print("剩余模板数:%d\t丢弃模板数:%d"%(count, miss_count))
    return templates

def read_pairs(pair_file):
    with open(pair_file, 'r') as f:
        pairs = f.readlines()
        pairs = [x.split('\n')[0] for x in pairs]
        pairs = [pair.split(',') for pair in pairs]
        pairs = [(int(pair[0]), int(pair[1])) for pair in pairs]
    return pairs

class IJBCTest:

    def __init__(self, file_list, index = None, thre=None):
        self.file_list = file_list
        self.subject_dict = build_subject_dict(file_list, index)
        self.verification_folds = None
        self.verification_templates = None
        self.verification_G1_templates = None
        self.verification_G2_templates = None
        self.thre = thre

    def init_verification_proto(self, protofolder):
        self.verification_folds = []
        self.verification_templates = []

        meta_gallery1 = os.path.join(protofolder,'ijbc_1N_gallery_G1.csv')
        meta_gallery2 = os.path.join(protofolder,'ijbc_1N_gallery_G2.csv')
        meta_probe = os.path.join(protofolder,'ijbc_1N_probe_mixed.csv')
        pair_file = os.path.join(protofolder,'ijbc_11_G1_G2_matches.csv')

        gallery_templates = build_templates(self.subject_dict, meta_gallery1, gallery=True, thre=self.thre)
        gallery_templates.extend(build_templates(self.subject_dict, meta_gallery2, gallery=True, thre=self.thre))
        gallery_templates.extend(build_templates(self.subject_dict, meta_probe, gallery=False, thre=self.thre))

        # Build pairs
        template_dict = {}
        for t in gallery_templates:
            template_dict[t.template_id] = t
        pairs = read_pairs(pair_file)
        self.verification_G1_templates = []
        self.verification_G2_templates = []
        drop_cout = 0
        remain_cout = 0
        for p in pairs:
            if p[0] in template_dict and p[1] in template_dict:
                self.verification_G1_templates.append(template_dict[p[0]])
                self.verification_G2_templates.append(template_dict[p[1]])
                remain_cout+=1
            else:
                drop_cout += 1
        print("丢弃:%d对，剩余：%d对"%(drop_cout, remain_cout))

        self.verification_G1_templates = np.array(self.verification_G1_templates, dtype=np.object)
        self.verification_G2_templates = np.array(self.verification_G2_templates, dtype=np.object)
    
        self.verification_templates = np.concatenate([
            self.verification_G1_templates, self.verification_G2_templates])
        print('{} templates are initialized.'.format(len(self.verification_templates)))


    def init_proto(self, protofolder):
        self.init_verification_proto(protofolder)

    def test_verification(self, compare_func, FARs=None):

        FARs = [1e-5, 1e-4, 1e-3, 1e-2] if FARs is None else FARs

        templates1 = self.verification_G1_templates
        templates2 = self.verification_G2_templates

        features1 = [t.feature for t in templates1]
        features2 = [t.feature for t in templates2]
        labels1 = np.array([t.label for t in templates1])
        labels2 = np.array([t.label for t in templates2])

        score_vec = compare_func(features1, features2)
        label_vec = labels1 == labels2

        tars, fars, thresholds = metrics.ROC(score_vec, label_vec, FARs=FARs)
        
        # There is no std for IJB-C
        std = [0. for t in tars]

        return tars, std, fars

