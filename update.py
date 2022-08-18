#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
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
        # image, label = self.dataset[self.idxs[item]]
        # return image, label


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        # self.trainloader, self.validloader = self.train_val_test(
        #     dataset, list(idxs))
        self.trainloader = self.train_split(dataset, idxs)  # dataset, and each element of dictionary
        # self.valilodaer = # validation self.
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 'cuda' if args.gpu else 'cpu'
        # Default criterion set to CrossEntropyLoss loss function
        # if args.loss == 'CrossEntropyLoss':
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_split(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)

        # idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        # idx[:3] get the first three elements
        idxss = list(idxs)
        idxs_train = idxss[:int(len(idxss))]
        # idxs_val = idxs[int(0.8 * len(idxss)):int(1.0 * len(idxss))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)), shuffle=False)  # int(len(idxs_val) / 10)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader  # , validloader #validloader  # testloader

    """
    # Normalisation des données audio
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(audio_train)
    audio_train_scaled = scaler.transform(audio_train)
    
    #====================================================================
    
    from sklearn import svm
    # Creating SVM with RBF kernel
    svc = svm.SVC(kernel='rbf', max_iter=-1, verbose = True, C=2.6)
    # Fitting SVM to all training data
    svc.fit(audio_train_scaled, y_train)
    # Prediction
    y_pred = svc.predict(scaler.transform(audio_test))
    """
    def update_weights(self, model, global_round):
        model.train()
        epoch_loss, epoch_accuracy = [], []
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        for iter in range(self.args.local_ep):
            batch_loss, batch_correct = [], []
            correct, total = 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # print(images.shape)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                # Inference
                _, pred_labels = torch.max(log_probs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round + 1, iter + 1, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                batch_correct.append(correct / total)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_accuracy.append(sum(batch_correct) / len(batch_correct))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(batch_correct) / len(batch_correct)

    # def update_weights(self, model, global_round):
    #     # Set mode to train model
    #     model.train()
    #     epoch_loss, epoch_accuracy = [], []
    #     # Set optimizer for the local updates
    #     if self.args.optimizer == 'sgd':
    #         optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)
    #     elif self.args.optimizer == 'adam':
    #         optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
    #     for iter in range(self.args.local_ep):
    #         batch_loss, batch_correct = [], []
    #         correct, total = 0.0, 0.0
    #         for batch_idx, (images, labels) in enumerate(self.trainloader):
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             # print(images.shape)
    #             model.zero_grad()
    #             log_probs = model(images)
    #             loss = self.criterion(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             # Inference
    #             _, pred_labels = torch.max(log_probs, 1)
    #             pred_labels = pred_labels.view(-1)
    #             correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #             total += len(labels)
    #             if self.args.verbose and (batch_idx % 10 == 0):
    #                 print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     global_round + 1, iter + 1, batch_idx * len(images),
    #                     len(self.trainloader.dataset),
    #                     100. * batch_idx / len(self.trainloader), loss.item()))
    #             self.logger.add_scalar('loss', loss.item())
    #             batch_loss.append(loss.item())
    #             batch_correct.append(correct / total)
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #         epoch_accuracy.append(sum(batch_correct) / len(batch_correct))
    #     return model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(batch_correct) / len(batch_correct)
    ######################
    # validate the model #
    ######################
    def inference_train(self, model):
        """
        Returns the inference accuracy and loss.
        """
        model.eval()
        with torch.no_grad():
            # total training labels, total training correct, validation loss,
            # total validation labels, total validation correct
            train_total, train_correct = 0.0, 0.0
            # train_pred_list = []
            # train_y_true = []
            for batch_idx, (images, labels) in enumerate(self.valiloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # Train Inference
                train_outputs = model(images)
                # Train prediction
                _, train_pred_labels = torch.max(train_outputs, 1)
                train_pred_labels = train_pred_labels.view(-1)
                train_correct += torch.sum(torch.eq(train_pred_labels, labels)).item()
                train_total += len(labels)
                # train_pred_list.extend(train_pred_labels.cpu().numpy())
                # train_y_true.extend(labels.cpu().numpy())
            train_acc = train_correct / train_total
            # fl_score = f1_score(train_y_true, train_pred_list, average=None)
        return train_acc  # , fl_score


class MultiLocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        # self.LAMBDA = 0.25
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.trainloader = self.train_split(dataset, idxs)
        # self.trainloader_CXR, self.trainloader_CUS = self.train_split(dataset_list, idxs_list)
        # self.list_src, self.list_tar = list(enumerate(self.trainloader_CXR)), list(enumerate(self.trainloader_CUS))
        # if args.loss == 'CrossEntropyLoss':
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    # compute mmd loss for x_source input(images) and x_target input(images)
    # def pw_loss(self, x_src, x_tar, src_label, target_label):
    # DA_loss.mix_rbf_mmd2(x_src, x_tar, [self.GAMMA])
    # mmdloss = DA_loss.kernel_mmd(x_src, x_tar).to(self.device)
    # return mmdloss
    # def mmd_loss(x_src, x_tar):
    # return DA_loss.csa_loss(x_src, x_tar, src_label, target_label).to(self.device)
    def train_split(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)

        # idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        # idx[:3] get the first three elements
        idxss = list(idxs)
        idxs_train = idxss[:int(len(idxss))]
        # idxs_val = idxs[int(0.8 * len(idxss)):int(1.0 * len(idxss))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)), shuffle=False)  # int(len(idxs_val) / 10)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader  # , validloader #validloader  # testloader

    # def train_split(self, dataset, idxs):
    #     """
    #     Returns train, validation and test dataloaders for a given dataset
    #     and user indexes.
    #     """
    #     idxss = list(idxs)
    #     idxs_train = idxss[:int(len(idxss))]
    #     # idxs_val = idxs[int(0.8 * len(idxss)):int(1.0 * len(idxss))]
    #     trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
    #                              batch_size=self.args.local_bs, shuffle=True)
    #     return trainloader#trainloader_CXR, trainloader_CUS
    def multi_update_weights(self, model, global_round):
        model.train()
        epoch_loss, epoch_accuracy = [], []
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        for iter in range(self.args.local_ep):
            batch_loss, batch_correct = [], []
            correct, total = 0.0, 0.0
            for batch_idx, (images, audios, labels) in enumerate(self.trainloader):
                images, audios, labels = images.to(self.device), audios.to(self.device), labels.to(self.device)
                # print(images.shape)
                model.zero_grad()
                log_probs = model(images, audios)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                # Inference
                _, pred_labels = torch.max(log_probs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round + 1, iter + 1, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                batch_correct.append(correct / total)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_accuracy.append(sum(batch_correct) / len(batch_correct))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(batch_correct) / len(batch_correct)
    # def multi_update_weights(self, model, global_round):
    #     # Set mode to train model
    #     model.train()
    #     epoch_loss, epoch_accuracy = [], []
    #     # Set optimizer for the local updates
    #     if self.args.optimizer == 'sgd':
    #         optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,  weight_decay=0.01)
    #     elif self.args.optimizer == 'adam':
    #         optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
    #     # elif self.args.optimizer == 'adadelta':
    #     #     optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    #     for iter in range(self.args.local_ep):
    #         batch_loss, batch_correct = [], []
    #         correct, total = 0.0, 0.0
    #         for batch_idx, (images, audios, labels) in enumerate(self.trainloader):
    #             images, audios, labels = images.to(self.device), audios.to(self.device), labels.to(self.device)
    #             model.zero_grad()
    #             log_probs = model(images, audios)
    #             loss = self.criterion(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             # Inference
    #             _, pred_labels = torch.max(log_probs, 1)
    #             pred_labels = pred_labels.view(-1)
    #             correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #             total += len(labels)
    #             if self.args.verbose and (batch_idx % 10 == 0):
    #                 print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     global_round + 1, iter + 1, batch_idx * len(images),
    #                     len(self.trainloader),
    #                     100. * batch_idx / len(self.trainloader), loss.data))
    #             self.logger.add_scalar('loss', loss.item())
    #             batch_loss.append(loss.item())
    #             batch_correct.append(correct / total)
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #         epoch_accuracy.append(sum(batch_correct) / len(batch_correct))
    #     return model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(batch_correct) / len(batch_correct)


def multi_test_inference(model, test_dataset):
    model.eval()
    with torch.no_grad():
        loss, total, correct = 0.0, 0.0, 0.0
        pred_list = []
        y_true = []
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        criterion = nn.CrossEntropyLoss().to(device)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        for batch_id, (images, audios, target) in enumerate(testloader):
            # _, (x_tar, y_target) = list_tar[batch_j]
            images, audios, target = images.to(device), audios.to(device), target.to(device)
            y_src = model(images, audios)
            batch_loss = criterion(y_src, target)
            loss += batch_loss.item()
            # Prediction
            _, pred_labels = torch.max(y_src, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, target)).item()
            total += len(target)
            pred_list.extend(pred_labels.cpu().numpy())
            y_true.extend(target.cpu().numpy())
        #     batch_j += 1
        #     if batch_j >= len(list_tar): batch_j = 0
        accuracy = correct / total
        loss = loss / len(testloader)
        score_f1 = f1_score(y_true, pred_list, average=None)
    return accuracy, loss, score_f1


def test_inference(model, test_dataset):
    """
    Returns the test accuracy and loss.
    """
    model.eval()
    with torch.no_grad():
        loss, total, correct = 0.0, 0.0, 0.0
        pred_list = []
        y_true = []
        # device = 'cuda' if args.gpu else 'cpu'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        criterion = nn.CrossEntropyLoss().to(device)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            pred_list.extend(pred_labels.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

        accuracy = correct / total
        loss = loss / len(testloader)
        score_f1 = f1_score(y_true, pred_list, average=None)
    return accuracy, loss, score_f1  # , auroc


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    # TP    predict and label are 1 at the same time
    tp = (y_true * y_pred).sum().to(torch.float32)
    # TN    predict and label are 0 at the same time
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    # FN    predict 0 label 1
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    # FP    predict 1 label 0
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    # f1_score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    # accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1.requires_grad = is_training
    return f1, acc
