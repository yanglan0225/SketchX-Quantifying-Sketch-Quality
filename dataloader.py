import torch
import os
import torch.utils.data as data
from torch.utils.data import DataLoader
import time
import json
import config
import numpy as np
import data_utils
import cv2
import random

class QuickDraw(data.Dataset):
    def __init__(self, type, config):

        self.config = config
        since = time.time()
        print('*'*10, 'start %s data loading'%type, '*'*10)

        # prepare data set
        self.max_length = config.max_length
        self.cls_listes = json.load(open(config.cls_json_file, 'r'))
        _, self.cls_to_idx, self.idx_to_cls = self.assign_cls_to_idx(json.load(open('/home/yl/cvpr22/cvpr22_1/quickdraw_all.json', 'r')))

        self.all_data = None # 0-column is label, 1-column is sketch sequence

        for cls_name in self.cls_listes:
            #print('loading  ', cls_name)
            cls_data = np.load(config.data_dir + cls_name + '.npz', encoding='latin1',
                               allow_pickle=True)[type]
            if type == 'train':
                idx = random.sample(list(range(0, 70000)), 7000)
                cls_data = cls_data[idx]
            if type == 'valid':
                cls_data = cls_data
            if type == 'test':
                cls_data = cls_data
            label = self.cls_to_idx[cls_name]
            label_array = np.ones((len(cls_data), 1)) * label

            label_data = np.c_[label_array, cls_data]

            if self.all_data is not None:
                self.all_data = np.r_[self.all_data, label_data]
            else:
                self.all_data = label_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        :param item:
        :return: label, trim_data, segment_embedding, padding_mask
        """
        data = self.all_data[item]
        label = data[0]
        ori_s3_data = data[1]

        bounds, stroke_num, abs_s3 = data_utils.get_statistics(ori_s3_data)

        norm_data = data_utils.rescale(abs_s3, bounds, img_size=None) # normalize to [0, 1]

        norm_s5_data = self.convert_s3_2_s5(norm_data)

        pad_data, pad_mask = self.trim(norm_s5_data)

        triplet_mask = self.triplet_mask(len(norm_data)+1)

        triplet_mask[0] = 0
        #label=238
        if len(norm_s5_data) > self.max_length:
            return label, pad_data, self.max_length, item
        else:
            return label, pad_data, len(norm_s5_data), item#triplet_mask


    def triplet_mask(self, seq_len):
        if seq_len+1 <= self.max_length:
            mini_mask = np.tril(np.ones((seq_len+1, seq_len+1)))
            pad_mask = np.zeros((self.max_length, self.max_length))
            pad_mask[:seq_len+1, :seq_len+1] = mini_mask
            pad_mask[0] = 0
            return 1 - pad_mask
        else:
            return 1 - np.tril(np.ones((self.max_length, self.max_length)))

    def generate_mask_data(self, norm_data, mask_prob):
        """
        output a masked data and mask with mask_prob
        :param norm_data:
        :param mask_prob:
        :return: masked data, mask
        """
        mask = [1 if random.random() < mask_prob else 0 for _ in range(len(norm_data))]

        mask_index = np.where(np.array(mask) == 1)[0]

        norm_data[mask_index] = np.array((0, 0, 0))

        return norm_data

    def trim(self, norm_data):
        """
        add an extra [CLS] at the beginning of data
        trim the data if its length is larger than max length
        pad the data if its length is shorter than max length
        generate an attention mask to tag which item is padding

        padding is True
        original data is False
        :param norm_data:
        :return: trimmed data, padding mask
        """
        ### np.array([0, 0, 1]).reshape(1, -1) is CLS input

        pad_data = np.zeros((self.max_length, 5))
        #pad_data[0] = [1, 1, 1, 1, 1]
        if len(norm_data) < self.max_length:
            padding_mask = np.ones((self.max_length))
            padding_mask[:len(norm_data)] = 0
            pad_data[:len(norm_data)] = norm_data
        else:
            pad_data = norm_data[:self.max_length]
            pad_data[-1, -1] = 1
            padding_mask = np.zeros((self.max_length))

        return pad_data, padding_mask

    def segment_embedding(self, trim_data):
        """
            tag each segment belongs to different strokes
            [0,0,0,1,1,1,1,1,0,0,0,0]
        """
        segment_emb = []
        cur_seg_flag = 0
        for segment in trim_data:
            if segment[-1] == 1:
                segment_emb.append(cur_seg_flag)
                cur_seg_flag = cur_seg_flag ^ 1 # cur segment osciallates between 0 and 1
            else:
                segment_emb.append(cur_seg_flag)

        while len(segment_emb) < self.max_length:
            segment_emb.append(cur_seg_flag)

        return np.array(segment_emb)

    def convert_s3_2_s5(self, norm_s3_data):
        s5_data = np.zeros((len(norm_s3_data), 5))
        s5_data[:, :2] = norm_s3_data[:, :2]
        s5_data[:, 2] = 1 - norm_s3_data[:, 2]
        s5_data[:, 3] = norm_s3_data[:, 2]
        s5_data[-1, 2:] = [0, 0, 1]
        return s5_data

    def pad_data(self, norm_s5_data):
        pad_data = np.zeros((self.max_length, 5))
        pad_mask = np.ones(self.max_length)
        if len(norm_s5_data) >= self.max_length:
            pad_data = norm_s5_data[:self.max_length]
            pad_data[-1, 2:] = [0, 0, 1]
            pad_mask[:] = 0
        else:
            pad_data[:len(norm_s5_data)] = norm_s5_data
            pad_mask[:len(norm_s5_data)] = 0

        return pad_data, pad_mask

    def assign_cls_to_idx(self, cls_list):
        """
        assign each classname (string) an int label
        :param cls_list: classname list
        :return: dict
        """
        cls_list.sort()
        cls_to_idx = {cls_list[i]: i for i in range(len(cls_list))}
        idx_to_cls = {str(i): cls_list[i] for i in range(len(cls_list))}
        return cls_list, cls_to_idx, idx_to_cls

def get_loader(config):
    train_set = QuickDraw('train', config)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_worker)
    max_train_batch = len(train_set)// config.batch_size + 1

    val_set = QuickDraw('valid', config)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    max_valid_batch = len(val_set) // config.batch_size + 1
    return train_loader, max_train_batch, val_loader, max_valid_batch

def evaluate_loader(config):
    val_set = QuickDraw('test', config)
    val_loader = DataLoader(val_set, batch_size=50, shuffle=False, num_workers=1, pin_memory=True)
    max_valid_batch = len(val_set) // config.batch_size + 1
    return val_set.idx_to_cls, val_loader, max_valid_batch

def idx_to_cls(cls_list):
        """
        assign each classname (string) an int label
        :param cls_list: classname list
        :return: dict
        """
        cls_list.sort()
        cls_to_idx = {cls_list[i]: i for i in range(len(cls_list))}
        idx_to_cls = {str(i): cls_list[i] for i in range(len(cls_list))}
        return idx_to_cls

if __name__=="__main__":

    config = config.Config()
    dataset = QuickDraw('valid', config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    idx_2_cls = idx_to_cls(json.load(open(config.cls_json_file, 'r')))

    save_dir = '/home/yl/quickdraw_raster_all/valid/'

    for i, data in enumerate(dataloader):
        cls_label = data[0]
        idx = data[1]
        idx = idx % 2500
        img_data = data[2]
        cls_name = idx_2_cls[str(int(cls_label.item()))]
        save_cls_path = save_dir + cls_name
        if not os.path.exists(save_cls_path):
            os.makedirs(save_cls_path)
        cv2.imwrite(save_cls_path + '/' +str(idx.item()) + '.png',
                    np.array(torch.squeeze(img_data, 0)))
