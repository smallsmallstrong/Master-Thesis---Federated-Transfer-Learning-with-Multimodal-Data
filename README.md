# Master-Thesis: Federated-Transfer-Learning-with-Multimodal-Data
This is the code for my master thesis. 

Federated Learning supports collecting a wealth of multimodal data from different devices without sharing raw data. Transfer Learning methods help transfer knowledge from some devices to others. Federated Transfer Learning methods benefit both Federated Learning and Transfer Learning.

Training in the new framework can be divided in three steps. In the first step, users who have data with the identical modalities are grouped together. For example, user with only sound signals are in group one, and those with only images are in group two, and users with multimodal data are in group three, and so on. In the second step, Federated Learning is executed within the groups, where Supervised Learning and Self-Supervised Learning are used depending on the group’s nature. Most of the Transfer Learning happens in the third step, where the related parts in the network obtained from the previous steps are aggregated (federated).

Part of the code for Federated Learning refer to https://github.com/AshwinRJ/Federated-Learning-PyTorch

Multimodal dataset is https://www.kaggle.com/datasets/birdy654/scene-classification-images-and-audio

If you feel my code is helpful, please give me a star and refer to my code.

Still need to change...

