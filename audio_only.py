import os
from pathlib import Path
import numpy as np
import pandas as pd

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

# # Normalisation des données audio
# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler().fit(train_audio)
# audio_train_scaled = scaler.transform(train_audio)
#
# #====================================================================
#
# from sklearn import svm
#
# # Creating SVM with RBF kernel
# svc = svm.SVC(kernel='rbf', max_iter=-1, verbose = True, C=2.6)
#
# # Fitting SVM to all training data
# svc.fit(audio_train_scaled, train_labels)
#
# # Prediction
# y_pred = svc.predict(scaler.transform(test_audio))
# print(y_pred.shape)
# print(y_pred)
#
# #====================================================================
# # Création du ficher de soumission
#
# submission = pd.DataFrame({'CLASS':y_pred})
# submission=submission.reset_index()
# submission = submission.rename(columns={'index': 'Id'})
#
# #======================================================================
# # Sauvegarde du fichier
# submission.to_csv('audio_submission_v3.csv', index=False)

### test for SVM
import torch
data_df_svm = pd.read_csv(os.path.join(datadir, 'submissionAmaniCaioTest1.csv'), delimiter=',', nrows=None)
pre_data = np.array(data_df_svm)
pre_labels = pre_data[:, -1].astype('int32')
def test_SVM(test_labels,pre_labels):
    correct = np.sum(np.equal(test_labels, pre_labels)).item()
    acc = correct/len(test_labels)
    return acc

accuracy = test_SVM(test_labels,pre_labels)
print(accuracy)