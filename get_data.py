#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7
"""
Get different datasets
"""
import copy
from pathlib import Path
import cv2
import numpy as np
import torch
import os
import pandas as pd
# import fasttext
# import json
from sklearn import preprocessing
from scene_dataset import ImageDataset, AudioDataset, ImageAudioDataset
from torchvision import datasets, transforms
from sampling import img_iid, img_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    class_list = {'FOREST': 0, 'CITY': 1, 'BEACH': 2, 'CLASSROOM': 3, 'RIVER': 4, 'JUNGLE': 5, 'RESTAURANT': 6,
                  'GROCERY-STORE': 7, 'FOOTBALL-MATCH': 8}
    datadir = Path.cwd() / 'data'
    if args.model == 'unimodal':
        if args.dataset == 'image':
            # train data
            data_df_train = pd.read_csv(os.path.join(datadir, 'train.csv'), delimiter=',', nrows=None)
            data_df_train['CLASS2'] = data_df_train['CLASS2'].map(class_list)
            train_data = np.array(data_df_train)
            train_img_list = data_df_train['IMAGE']
            train_labels = train_data[:, -1].astype('int32')
            # vali data
            data_df_vali = pd.read_csv(os.path.join(datadir, 'vali.csv'), delimiter=',', nrows=None)
            data_df_vali['CLASS2'] = data_df_vali['CLASS2'].map(class_list)
            vali_data = np.array(data_df_vali)
            vali_img_list = data_df_vali['IMAGE']
            vali_labels = vali_data[:, -1].astype('int32')
            # test data
            data_df_test = pd.read_csv(os.path.join(datadir, 'test.csv'), delimiter=',', nrows=None)
            data_df_test['CLASS2'] = data_df_test['CLASS2'].map(class_list)
            test_data = np.array(data_df_test)
            test_img_list = data_df_test['IMAGE']
            test_labels = test_data[:, -1].astype('int32')
            train_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
            test_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            train_dataset = ImageDataset(root_dir=datadir,
                            files=train_img_list,
                            labels=train_labels,
                            img_transform=train_transform)
            vali_dataset = ImageDataset(root_dir=datadir,
                            files=vali_img_list,
                            labels=vali_labels,
                            img_transform=test_transform)
            test_dataset = ImageDataset(root_dir=datadir,
                            files=test_img_list,
                            labels=test_labels,
                            img_transform=test_transform)
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = img_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = img_noniid(train_dataset, args.num_users)
        if args.dataset == 'audio':
            # train data
            data_df_train = pd.read_csv(os.path.join(datadir, 'train.csv'), delimiter=',', nrows=None)
            data_df_train['CLASS2'] = data_df_train['CLASS2'].map(class_list)
            train_data = np.array(data_df_train)
            train_audio = train_data[:, 1:-2].astype('float32')
            train_labels = train_data[:, -1].astype('int32')
            scaler_train = preprocessing.StandardScaler().fit(train_audio)
            audio_train_scaled = scaler_train.transform(train_audio)
            # vali data
            data_df_vali = pd.read_csv(os.path.join(datadir, 'vali.csv'), delimiter=',', nrows=None)
            data_df_vali['CLASS2'] = data_df_vali['CLASS2'].map(class_list)
            vali_data = np.array(data_df_vali)
            vali_audio = vali_data[:, 1:-2].astype('float32')
            vali_labels = vali_data[:, -1].astype('int32')
            scaler_vali = preprocessing.StandardScaler().fit(vali_audio)
            audio_vali_scaled = scaler_vali.transform(vali_audio)
            # test data
            data_df_test = pd.read_csv(os.path.join(datadir, 'test.csv'), delimiter=',', nrows=None)
            data_df_test['CLASS2'] = data_df_test['CLASS2'].map(class_list)
            test_data = np.array(data_df_test)
            test_audio = test_data[:, 1:-2].astype('float32')
            test_labels = test_data[:, -1].astype('int32')
            audio_transform = None
            scaler_test = preprocessing.StandardScaler().fit(test_audio)
            audio_test_scaled = scaler_test.transform(test_audio)
            train_dataset = AudioDataset(root_dir=datadir,
                                         audio=audio_train_scaled,
                                         labels=train_labels,
                                         audio_transform=audio_transform)
            vali_dataset = AudioDataset(root_dir=datadir,
                                        audio=audio_vali_scaled,
                                        labels=vali_labels,
                                        audio_transform=audio_transform)
            test_dataset = AudioDataset(root_dir=datadir,
                                        audio=audio_test_scaled,
                                        labels=test_labels,
                                        audio_transform=audio_transform)
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = img_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = img_noniid(train_dataset, args.num_users)

    # print('user_groups', user_groups)
    return train_dataset, vali_dataset, test_dataset, user_groups


def get_multi_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    class_list = {'FOREST': 0, 'CITY': 1, 'BEACH': 2, 'CLASSROOM': 3, 'RIVER': 4, 'JUNGLE': 5, 'RESTAURANT': 6,
                  'GROCERY-STORE': 7, 'FOOTBALL-MATCH': 8}
    datadir = Path.cwd() / 'data'
    if args.model == 'multimodal':
        if args.dataset == 'image_audio':
            # train data
            data_df_train = pd.read_csv(os.path.join(datadir, 'train.csv'), delimiter=',', nrows=None)
            data_df_train['CLASS2'] = data_df_train['CLASS2'].map(class_list)
            train_data = np.array(data_df_train)
            train_img_list = data_df_train['IMAGE']
            train_audio = train_data[:, 1:-2].astype('float32')
            train_labels = train_data[:, -1].astype('int32')
            scaler_train = preprocessing.StandardScaler().fit(train_audio)
            audio_train_scaled = scaler_train.transform(train_audio)
            # vali data
            data_df_vali = pd.read_csv(os.path.join(datadir, 'vali.csv'), delimiter=',', nrows=None)
            data_df_vali['CLASS2'] = data_df_vali['CLASS2'].map(class_list)
            vali_data = np.array(data_df_vali)
            vali_img_list = data_df_vali['IMAGE']
            vali_audio = vali_data[:, 1:-2].astype('float32')
            vali_labels = vali_data[:, -1].astype('int32')
            scaler_vali = preprocessing.StandardScaler().fit(vali_audio)
            audio_vali_scaled = scaler_vali.transform(vali_audio)
            # test data
            data_df_test = pd.read_csv(os.path.join(datadir, 'test.csv'), delimiter=',', nrows=None)
            data_df_test['CLASS2'] = data_df_test['CLASS2'].map(class_list)
            test_data = np.array(data_df_test)
            test_img_list = data_df_test['IMAGE']
            test_audio = test_data[:, 1:-2].astype('float32')
            test_labels = test_data[:, -1].astype('int32')
            scaler_test = preprocessing.StandardScaler().fit(test_audio)
            audio_test_scaled = scaler_test.transform(test_audio)
            train_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
            test_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            audio_transform = None
            train_dataset = ImageAudioDataset(root_dir=datadir,
                                              files=train_img_list,
                                              audio=audio_train_scaled,
                                              labels=train_labels,
                                              img_transform=train_transform,
                                              audio_transform=audio_transform)
            vali_dataset = ImageAudioDataset(root_dir=datadir,
                                             files=vali_img_list,
                                             audio=audio_vali_scaled,
                                             labels=vali_labels,
                                             img_transform=test_transform,
                                             audio_transform=audio_transform)
            test_dataset = ImageAudioDataset(root_dir=datadir,
                                             files=test_img_list,
                                             audio=audio_test_scaled,
                                             labels=test_labels,
                                             img_transform=test_transform,
                                             audio_transform=audio_transform)
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = img_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = img_noniid(train_dataset, args.num_users)

        return train_dataset, vali_dataset, test_dataset, user_groups # may be need vali dataset


def get_frames(filename, n_frames=1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len - 1, n_frames + 1, dtype=np.int16)
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


from torch.utils.data import DataLoader, Dataset
class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]  # idx is a list type

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        sample = self.dataset[self.idxs[item]]
        return sample

if __name__ == "__main__":
    from options import args_parser
    from models import Audio_svm, vgg_16,mymodel2
    args = args_parser()
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    train_dataset, vali_dataset, test_dataset, user_groups = get_multi_dataset(args)
    # svc = Audio_svm()
    from sklearn import svm
    # model = vgg_16(flag=False)
    model = mymodel2()
    # Creating SVM with RBF kern
    # svc = svm.SVC(kernel='rbf', max_iter=-1, verbose=True, C=2.6)
    for idx in idxs_users:
        # idxs_train = idxss[:int(len(idxss))]
        trainloader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                             batch_size=10, shuffle=True)
        # print(len(trainloader))
        for index, data in enumerate(trainloader):
            image, audio, label = data
            # outputs = svc(audio)
            # outputs = svc.fit(audio, label)
            outputs = model(image, audio)
            # print(outputs.shape)
            # svc.fit(audio, label)
            print(image.shape)
            print(audio.shape)

            # print(data['image'].shape)
            # print(data['text'][0])
            # print(data['text'][1])
            # print(data['label'])

