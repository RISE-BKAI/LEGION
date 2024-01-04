import os,sys
import torch
import pickle
import numpy as np
from torch import nn
import pandas as pd 
from transformers import * 
from tqdm import trange
import pickle

source_dir = './'
suffix = 'rand123'
model_name = 'bert_base'
model_para = "models/bert_base_DBloss_rand123_epoch18para"
model_checkpoint = os.path.join('berts', 'bert-base-uncased')
    
data_test=pickle.load(open(os.path.join(source_dir, 'data', 'data_test.'+suffix),'rb'))
labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
num_labels = len(labels_ref)
max_len = 512
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, max_len=max_len)#, use_fast=True)
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)).to(device)

from torch.utils.data import Dataset, DataLoader

def preprocess_function(docu):
    labels = [1 if x in docu['labels'] else 0 for x in labels_ref] 
    encodings = tokenizer(docu['text'], truncation=True, padding='max_length')    
    return (torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']), torch.tensor(labels))

class CustomDataset(Dataset):
    '''Characterizes a dataset for PyTorch'''
    def __init__(self, documents):
        '''Initialization'''
        self.documents = documents

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.documents)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        return preprocess_function(self.documents[index])

test_dataloader = DataLoader(CustomDataset(data_test), shuffle=False, batch_size=batch_size)  

group_head_and_mid = [i for i, x in enumerate(class_freq) if x>4]

model.load_state_dict(torch.load(model_para))
model.eval()

pred_labels = []
for i, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
        # Forward pass
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

            predictions = [np.argsort(row)[-5:] for row in pred_label]
            predictions = [row[::-1] for row in predictions]
            predictions = [[p for p in row if p in group_head_and_mid] for row in predictions]

            for n in range(len(pred_label)):
                for t in range(len(pred_label[n])):
                    if t not in group_head_and_mid or pred_label[n][t]<0.55:
                        pred_label[n][t] = 0

            for n in range(len(pred_label)):
                for t in range(len(pred_label[n])):
                    if t not in predictions[n]:
                        pred_label[n][t] = 0
        pred_labels.append(pred_label)

pred_labels_test = [item for sublist in pred_labels for item in sublist]
print(np.shape(pred_labels_test))
print(pred_labels_test[0])

pickle.dump(pred_labels_test, open('data/pred_labels_test_head_mid','wb')) 
df = pd.DataFrame(data=pred_labels_test)
# test
pred_labels_test=pickle.load(open('data/pred_labels_test_head_mid','rb'))
print(type(pred_labels_test))
print(np.shape(pred_labels_test))