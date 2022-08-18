import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, AdamW


class TextDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = pd.read_json(data_path, lines=True)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "text"]
        label = self.dataset.loc[idx, "label"]
        sample = {"text": text, "label": label}
        return sample
train_dataset = TextDataset('data/train.jsonl')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
lr = 1e-5
global_model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",  # 12-layer BERT model, with an uncased vocab.
                num_labels=2,  # Binary classification
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
bert_optimizer = AdamW(global_model.parameters(), lr=0.00001)
crit = torch.nn.CrossEntropyLoss()

epoch =10
for e in range(10):
    epoch_loss, epoch_acc = 0., 0.
    total_len = 0
    for i, data in enumerate(train_loader):
        global_model.train()
        sentence = data['text']
        label = data['label']
        predict = global_model(sentence)
        loss = crit(predict, label)
        epoch_loss += loss.item()
        epoch_acc += predict.eq(label.data.view_as(predict)).cpu().sum()
        total_len += len(label)
        bert_optimizer.zero_grad()
        loss.backward()
        bert_optimizer.step()

    print('epoch_'+e,'acc_'+epoch_acc/total_len,'loss_'+epoch_loss/total_len)