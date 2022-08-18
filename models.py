
"""
If we use deep-CORAL (transfer learning)
"""

import torch
import torch.nn as nn
import torchvision
# from sklearn import svm

"""
#====================================================================
# 2. Hyperparameter optimization 
https://github.com/caiolang/Multimodal-Classification/blob/main/audio_only.ipynb 
# -----------------------------------------------------------------------------------------------------------------------------------------------
# Attempt to optimize hyperparameters "manually", checking the precision on train and test sets for different values of C (regularity of the SVM)
# -----------------------------------------------------------------------------------------------------------------------------------------------

from sklearn import svm
C_test = np.logspace(-1,1.2,15)

for i in C_test:
    svc = svm.SVC(kernel='rbf', max_iter=-1, verbose = True, C=i)
    svc.fit(Xn, yn)
    score_train = svc.score(Xn, yn)
    score_test = svc.score(Xv, yv)
    
# A good value apparently is C=2.6
# Good values must have high precision on both train and test sets (low variance),
 without a great difference between them(low bias)
"""
class Audio_mlp(nn.Module):
    def __init__(self, n_feature=104, n_class=9):
        super(Audio_mlp, self).__init__()
        self.hidden1 = nn.Linear(n_feature, 977)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(977, 365)
        self.relu2 = nn.ReLU()
        self.hidden3 = nn.Linear(365, 703)
        self.relu3 = nn.ReLU()
        self.hidden4 = nn.Linear(703, 41)
        self.relu4 = nn.ReLU()
        self.fc = nn.Linear(41, n_class)

    def forward(self, x):
        h1 = self.hidden1(x)
        h1 = self.relu1(h1)
        h2 = self.hidden2(h1)
        h2 = self.relu2(h2)
        h3 = self.hidden3(h2)
        h3 = self.relu3(h3)
        h4 = self.hidden4(h3)
        h4 = self.relu4(h4)
        out = self.fc(h4)
        return out


# audio-only, creating SVM with RBF kernel, with the best hyperparameter optimization
class Audio_svm(nn.Module):
    # svc = svm.SVC(kernel='rbf', max_iter=-1, verbose=True, C=2.6)
    # return svc
    def __init__(self, n_feature=104, n_class=9):
        super(Audio_svm, self).__init__()
        self.fc = nn.Linear(n_feature, n_class)
        # torch.nn.init.kaiming_uniform_(self.fc.weight)
        # torch.nn.init.constant_(self.fc.bias, 0.1)

    def forward(self, x):
        output = self.fc(x)
        return output

def audio_svm():
    svm = Audio_svm()
    return svm

def audio_mlp():
    mlp = Audio_mlp()
    return mlp

def vgg_16(flag=True):
    vgg16_model = torchvision.models.vgg16(pretrained=flag)
    return vgg16_model


def resnet_18(flag=True):
    resnet = torchvision.models.resnet18(pretrained=flag)
    return resnet

# Late fusion for multi-layer perceptron
# class MLP_fusion(nn.Module):
#     def __init__(self):
#         super(MLP_fusion, self).__init__()
#         # VGG16 for images
#         self.vgg16 = torchvision.models.vgg16(pretrained=True)
#         num_ftrs = self.vgg16.classifier[6].in_features
#         self.vgg16.classifier[6] = nn.Linear(num_ftrs, 41)
#         # D-MLP for audios
#         self.hidden1 = nn.Linear(104, 977)
#         self.relu1 = nn.ReLU()
#         self.hidden2 = nn.Linear(977, 365)
#         self.relu2 = nn.ReLU()
#         self.hidden3 = nn.Linear(365, 703)
#         self.relu3 = nn.ReLU()
#         self.hidden4 = nn.Linear(703, 41)
#         # self.relu4 = nn.ReLU()
#         self.ac1 = nn.ReLU()
#         self.fc1 = nn.Linear(82, 9)
#
#     def forward(self, img, audio):
#         x1 = self.vgg16(img)
#         x2_h1 = self.hidden1(audio)
#         x2_h1 = self.relu1(x2_h1)
#         x2_h2 = self.hidden2(x2_h1)
#         x2_h2 = self.relu2(x2_h2)
#         x2_h3 = self.hidden3(x2_h2)
#         x2_h3 = self.relu3(x2_h3)
#         x2 = self.hidden4(x2_h3)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.ac1(x)
#         x = self.fc1(x)
#         return x

class MLP_fusion(nn.Module):
    def __init__(self):
        super(MLP_fusion, self).__init__()
        # VGG16 for images
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        num_ftrs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_ftrs, 703)  # num_ftrs 4096
        self.relu_vgg = nn.ReLU()
        self.vgg_mlp = nn.Linear(703, 41)
        self.relu_mlp = nn.ReLU()
        # D-MLP for audios
        self.hidden1 = nn.Linear(104, 977)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(977, 365)
        self.relu2 = nn.ReLU()
        self.hidden3 = nn.Linear(365, 703)
        self.relu3 = nn.ReLU()
        self.hidden4 = nn.Linear(703, 41)
        self.relu4 = nn.ReLU()
        # self.ac1 = nn.ReLU()
        self.fc1 = nn.Linear(41, 9)

    def forward(self, img, audio):
        x1 = self.vgg16(img)
        x1 = self.relu_vgg(x1)
        x1 = self.vgg_mlp(x1)
        x1 = self.relu_mlp(x1)

        x2_h1 = self.hidden1(audio)
        x2_h1 = self.relu1(x2_h1)
        x2_h2 = self.hidden2(x2_h1)
        x2_h2 = self.relu2(x2_h2)
        x2_h3 = self.hidden3(x2_h2)
        x2_h3 = self.relu3(x2_h3)
        x2_h4 = self.hidden4(x2_h3)
        x2 = self.relu4(x2_h4)
        x_concat = torch.cat((x1, x2), dim=1)  # the combined features
        # x = self.ac1(x_concat)
        x = self.fc1(x_concat)
        return x


class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()
        # transfer learning - pretrained model
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        num_ftrs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_ftrs, 20)
        # print(self.vgg16.classifier[6])
        self.fc1 = nn.Linear(104, 10)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(30, 9)
        # print(self.vgg16)
        # print(self.resnext)

    def forward(self, img, audio):
        # x1 = self.resnext(img)
        x1 = self.vgg16(img)
        # x1 = self.vgg16.fc(x1)
        # print(x1.shape)

        x2 = self.fc1(audio)
        # print(x2.shape)
        x = torch.cat((x1, x2), dim=1)
        x = self.ac1(x)
        x = self.fc2(x)
        return x

def mymodel2():
    mymodel = MyModel2()
    return mymodel

def mlp_fusion():
    mlp_fuse = MLP_fusion()
    return mlp_fuse
# model = mymodel2()
# print(model)

"""
vgg16
(fc): Linear(in_features=4096, out_features=20, bias=True)
(fc1): Linear(in_features=104, out_features=10, bias=True)
(ac1): ReLU()
(fc2): Linear(in_features=30, out_features=9, bias=True)
----------------------------------------------------------
resnet
(fc): Linear(in_features=512, out_features=20, bias=True)
(fc1): Linear(in_features=104, out_features=10, bias=True)
(ac1): ReLU()
(fc2): Linear(in_features=30, out_features=9, bias=True)
"""
# def total_param(l=[]):
#     s = 0
#     for i in range(len(l)-1):
#         s = s+l[i]*l[i+1]+l[i+1]
#     return s
#
# tos = total_param([104,977,365,703,41,9])
# print(tos)
