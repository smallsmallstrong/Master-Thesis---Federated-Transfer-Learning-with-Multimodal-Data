import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scene_dataset import ImageDataset, AudioDataset, ImageAudioDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from models import mymodel2
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
np.random.seed(0)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class_list = {'FOREST': 0, 'CITY': 1, 'BEACH': 2, 'CLASSROOM': 3, 'RIVER': 4, 'JUNGLE': 5, 'RESTAURANT': 6,
              'GROCERY-STORE': 7, 'FOOTBALL-MATCH': 8}
datadir = Path.cwd() / 'data'
# TRAIN DATA
data_df_train = pd.read_csv(os.path.join(datadir, 'train.csv'), delimiter=',', nrows=None)
data_df_train['CLASS2'] = data_df_train['CLASS2'].map(class_list)
train_data = np.array(data_df_train)
train_img_list = data_df_train['IMAGE']
train_audio = train_data[:, 1:-2].astype('float32')
train_labels = train_data[:, -1].astype('int32')
# TEST DATA
data_df_test = pd.read_csv(os.path.join(datadir, 'test.csv'), delimiter=',', nrows=None)
data_df_test['CLASS2'] = data_df_test['CLASS2'].map(class_list)
test_data = np.array(data_df_test)
test_img_list = data_df_test['IMAGE']
test_audio = test_data[:, 1:-2].astype('float32')
test_labels = test_data[:, -1].astype('int32')

model = mymodel2()
# Configurations de l'apprentissage
softmax = torch.nn.Softmax(dim=1)
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
epochs = 10
nsample = 10
train_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
audio_transform = None
train_dataset = ImageAudioDataset(root_dir=datadir,
                                              files=train_img_list,
                                              audio=train_audio,
                                              labels=train_labels,
                                              img_transform=train_transform,
                                              audio_transform=audio_transform)
datafull_loader = DataLoader(train_dataset, batch_size=nsample, shuffle=True)

for epoch in range(epochs):
    model.train()
    print("epoch", epoch)
    for data in datafull_loader:
        img, audio, label = data

        # if using a model that has audio and image as inputs
        img, audio, label = img.to(device), audio.to(device), label.to(device)
        output = model(img, audio)

        # if using a model that has only image as input
        #         img, label = img.to(device), label.to(device)
        #         output = model(img)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if np.random.randint(0, 20) == 0:
            print("\tloss=", loss)  ## on affiche pour valider que ça diverge pas

from tqdm import tqdm

## Taille du batch
nsample = 64

test_dataset = ImageAudioDataset(root_dir=datadir,
                                files=test_img_list,
                                audio=test_audio,
                                labels=None,
                                img_transform=train_transform,
                                audio_transform=None)

datatest_loader = DataLoader(test_dataset, batch_size=nsample,shuffle=False)

pred_labels = []

with torch.no_grad():
    model.eval()
    for i,data in enumerate(tqdm(datatest_loader)):
        img,audio = data
        img,audio = img.to(device), audio.to(device)
#         output = model(img)
        output = model(img, audio)
        y_pred = softmax(output)
        label_pred = torch.argmax(y_pred,dim=1)
        for label in label_pred:
            pred_labels.append(int(label))

print(pred_labels)

# import matplotlib.pyplot as plt
# from PIL import Image
# # visu image
# idx = 56
# class_list = ['FOREST', 'CITY', 'BEACH', 'CLASSROOM', 'RIVER', 'JUNGLE', 'RESTAURANT', 'GROCERY-STORE', 'FOOTBALL-MATCH']
# img = Image.open(os.path.join(datadir, test_img_list.iloc[idx]))
# plt.imshow(np.asarray(img))

# print("number of images:", test_img_list.shape[0])
# print("shape of each image:", np.asarray(img).shape)
# print("this is TEST image number", idx)
# print("predicted class:", class_list[pred_labels[idx]])

submission = pd.DataFrame({'CLASS': pred_labels})
submission=submission.reset_index()
submission = submission.rename(columns={'index': 'Id'})

print("number of images:",len(submission))
submission.to_csv('submissionAmaniCaioTest.csv',index=False)