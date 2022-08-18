#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7
"""
Get different datasets
"""
"""
search for papers using the same ultrasound videos
What is the model they use?
How do they pre-process their data?
Give F1 scores for your current state to make it comparable to values in the paper
Try to improve CT and Ultrasound with VGG19
Using sliding windows from Ultrasound video instead of sampling
"""

import copy
import cv2
import numpy as np
import torch
from torchvision import datasets, transforms

# import transformcustom
from sampling import CT_iid, CT_noniid
from sampling import CUS_iid, CUS_noniid
# from sampling import CUS_video_iid
# from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import CXR_iid, CXR_noniid
# from videocustom import VideoLabelDataset
from videosets import VideoFrameDataset, ImglistToTensor


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.model == 'unimodal':
        if args.dataset == 'CXR':
            train_transform = transforms.Compose([
                # transforms.Grayscale(),
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
            test_transform = transforms.Compose([
                # transforms.Grayscale(),
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # transforms.Normalize((0.5,), (0.5,)),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train_dataset = datasets.ImageFolder('data/CXR/train', transform=train_transform)
            test_dataset = datasets.ImageFolder('data/CXR/test', transform=test_transform)
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = CXR_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = CXR_noniid(train_dataset, args.num_users)

        if args.dataset == 'CT_LSTM':
            # data_dir = 'data/CT/'
            train_transform = transforms.Compose([
                ImglistToTensor(),
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
            test_transform = transforms.Compose([
                ImglistToTensor(),
                transforms.Resize(size=(224, 224)),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            train_dataset = VideoFrameDataset(
                root_path='data/CT/train',
                annotationfile_path='data/CT/train/annotations.txt',
                num_segments=6,
                frames_per_segment=1,
                imagefile_template='IM{:05d}.png',
                transform=train_transform,
                random_shift=True,
                test_mode=False)
            vali_dataset = VideoFrameDataset(
                root_path='data/CT/vali',
                annotationfile_path='data/CT/vali/annotations.txt',
                num_segments=6,
                frames_per_segment=1,
                imagefile_template='IM{:05d}.png',
                transform=test_transform,
                random_shift=False,
                test_mode=True
            )
            test_dataset = VideoFrameDataset(
                root_path='data/CT/test',
                annotationfile_path='data/CT/test/annotations.txt',
                num_segments=6,
                frames_per_segment=1,
                imagefile_template='IM{:05d}.png',
                transform=test_transform,
                random_shift=False,
                test_mode=True
            )
            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = CT_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = CT_noniid(train_dataset, args.num_users)

        if args.dataset == 'CT':
            train_transform = transforms.Compose([
                # transforms.Grayscale(),
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
            test_transform = transforms.Compose([
                # transforms.Grayscale(),
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # transforms.Normalize((0.5,), (0.5,)),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train_dataset = datasets.ImageFolder('data/CT/train', transform=train_transform)
            test_dataset = datasets.ImageFolder('data/CT/test', transform=test_transform)
            vali_dataset = datasets.ImageFolder('data/CT/vali', transform=test_transform)
            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = CT_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = CT_noniid(train_dataset, args.num_users)

        if args.dataset == 'CUS_LSTM':
            train_transform = transforms.Compose([
                ImglistToTensor(),
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(10, translate=(0.1, 0.1)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
            test_transform = transforms.Compose([
                ImglistToTensor(),
                transforms.Resize(size=(224, 224)),
                # transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train_dataset = VideoFrameDataset(
                root_path='data/CUS_3/train',
                annotationfile_path='data/CUS_3/train/annotations.txt',
                num_segments=10,
                frames_per_segment=1,
                imagefile_template='frame_{:05d}.jpg',
                transform=train_transform,
                random_shift=True,
                test_mode=False)
            vali_dataset = VideoFrameDataset(
                root_path='data/CUS_3/vali',
                annotationfile_path='data/CUS_3/vali/annotations.txt',
                num_segments=10,
                frames_per_segment=1,
                imagefile_template='frame_{:05d}.jpg',
                transform=test_transform,
                random_shift=False,
                test_mode=True)
            test_dataset = VideoFrameDataset(
                root_path='data/CUS_3/test',
                annotationfile_path='data/CUS_3/test/annotations.txt',
                num_segments=10,
                frames_per_segment=1,
                imagefile_template='frame_{:05d}.jpg',
                transform=test_transform,
                random_shift=False,
                test_mode=True
            )
            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = CUS_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = CUS_noniid(train_dataset, args.num_users)

        if args.dataset == 'CUS_3':
            train_transform = transforms.Compose([
                # transforms.Grayscale(),
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
            test_transform = transforms.Compose([
                # transforms.Grayscale(),
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # transforms.Normalize((0.5,), (0.5,)),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train_dataset = datasets.ImageFolder('data/CUS_3/train', transform=train_transform)
            test_dataset = datasets.ImageFolder('data/CUS_3/test', transform=test_transform)
            vali_dataset = datasets.ImageFolder('data/CUS_3/vali', transform=test_transform)
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = CUS_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = CUS_noniid(train_dataset, args.num_users)

    return train_dataset, vali_dataset, test_dataset, user_groups


def get_multi_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.model == 'multimodal':
        train_list, test_list, user_groups_list = [], [], []
        if args.dataset == 'CXR4CT':
            CXR_train_transform = transforms.Compose([
                # transforms.Grayscale(),
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
            CXR_test_transform = transforms.Compose([
                # transforms.Grayscale(),
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train_transform = transforms.Compose([
                ImglistToTensor(),
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
            test_transform = transforms.Compose([
                ImglistToTensor(),
                transforms.Resize(size=(224, 224)),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train_list[0] = datasets.ImageFolder('data/CXR/train', transform=CXR_train_transform)
            test_list[0] = datasets.ImageFolder('data/CXR/test', transform=CXR_test_transform)
            train_list[1] = VideoFrameDataset(
                root_path='data/CT/train',
                annotationfile_path='data/CT/train/annotations.txt',
                num_segments=5,
                frames_per_segment=1,
                imagefile_template='IM{:05d}.png',
                transform=train_transform,
                random_shift=True,
                test_mode=False)
            test_list[1] = VideoFrameDataset(
                root_path='data/CT/test',
                annotationfile_path='data/CT/test/annotations.txt',
                num_segments=5,
                frames_per_segment=1,
                imagefile_template='IM{:05d}.png',
                transform=test_transform,
                random_shift=False,
                test_mode=True
            )

            if args.iid:
                # Sample IID user data from Mnist
                user_groups_list[0] = CXR_iid(train_list[0], args.num_users)
                user_groups_list[1] = CT_iid(train_list[1], args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups_list[0] = CXR_noniid(train_list[0], args.num_users)
                    user_groups_list[1] = CXR_noniid(train_list[1], args.num_users)

    return train_list, test_list, user_groups_list


# def get_multi_dataset():

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


# def store_frames(frames, path2store):
#     for ii, frame in enumerate(frames):
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         path2img = os.path.join(path2store, "frame"+str(ii)+".jpg")
#         cv2.imwrite(path2img, frame)

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


import torch.nn as nn


class DaNN(nn.Module):
    def __init__(self, n_input=28 * 28, n_hidden=256, n_class=10):
        super(DaNN, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(n_hidden, n_class)

    def forward(self, src, tar):
        x_src = self.layer_input(src)
        x_tar = self.layer_input(tar)
        x_src = self.dropout(x_src)
        x_tar = self.dropout(x_tar)
        x_src_mmd = self.relu(x_src)
        x_tar_mmd = self.relu(x_tar)
        y_src = self.layer_hidden(x_src_mmd)
        return y_src, x_src_mmd, x_tar_mmd


if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = vgg19lstm(pretrained=False, in_channels=3, num_classes=4, dr_rate=0.1).to(device)
    # print(model)
    # N = 3 (Mini batch size)
    model = DaNN(n_input=28 * 28, n_hidden=256, n_class=10)
    data_src = torch.randn(1, 3, 3, 28, 28)
    data_tar = torch.randn(1, 3, 3, 28, 28)
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))
    batch_j = 0
    for batch_id, datas in enumerate(data_src):
        data, target = datas
        _, (x_tar, y_target) = list_tar[batch_j]
        data = data.data.view(-1, 28 * 28)
        print(data.shape)
        x_tar = x_tar.view(-1, 28 * 28)
        model.train()
        y_src, x_src_mmd, x_tar_mmd = model(data, x_tar)
