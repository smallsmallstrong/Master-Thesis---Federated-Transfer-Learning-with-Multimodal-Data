import os

import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_dir, files, labels=None, img_transform=None):
        self.root_dir = root_dir
        self.files = files
        # self.audio = audio
        self.labels = labels
        self.img_transform = img_transform
        # self.audio_transform = audio_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.files.iloc[idx]))
        # audio = self.audio[idx, :]
        if self.img_transform is not None:
            img = self.img_transform(img)
        # if self.audio_transform is not None:
        #     audio = self.audio_transform(audio)
        if self.labels is not None:
            return img, int(self.labels[idx])
        else:
            return img


class AudioDataset(Dataset):
    def __init__(self, root_dir, audio, labels=None, audio_transform=None):
        self.root_dir = root_dir
        # self.files = files
        self.audio = audio
        self.labels = labels
        # self.img_transform = img_transform
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        # img = Image.open(os.path.join(self.root_dir, self.files.iloc[idx]))
        audio = self.audio[idx, :]
        # if self.img_transform is not None:
        #     img = self.img_transform(img)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)
        if self.labels is not None:
            return audio, int(self.labels[idx])
        else:
            return audio


class ImageAudioDataset(Dataset):
    def __init__(self, root_dir, files, audio, labels, img_transform, audio_transform):
        self.root_dir = root_dir
        self.files = files
        self.audio = audio
        self.labels = labels
        self.img_transform = img_transform
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.files.iloc[idx]))
        audio = self.audio[idx, :]
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)
        if self.labels is not None:
            return img, audio, int(self.labels[idx])
        else:
            return img, audio


# # ----------------Test-----------------
# img_list_transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# audio_transform = None
# datadir = 'data'
# data_df = pd.read_csv(os.path.join(datadir, 'dataset.csv'), delimiter=',', nrows=None)
# class_list = {'FOREST': 0, 'CITY': 1, 'BEACH': 2, 'CLASSROOM': 3, 'RIVER': 4, 'JUNGLE': 5, 'RESTAURANT': 6,
#               'GROCERY-STORE': 7, 'FOOTBALL-MATCH': 8}
# data_df['CLASS2'] = data_df['CLASS2'].map(class_list)
# data = np.array(data_df)
# img_list = data_df['IMAGE']
# audio = data[:, 1:-2].astype('float32')
# labels = data[:, -1].astype('int32')
# dataset = ImageAudioDataset(root_dir=datadir,
#                             files=img_list,
#                             audio=audio,
#                             labels=labels,
#                             img_transform=img_list_transform,
#                             audio_transform=audio_transform)
#
# trainloader = DataLoader(dataset, batch_size=10, shuffle=True)
# # print(len(trainloader))
