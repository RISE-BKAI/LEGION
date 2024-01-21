import os,sys
import torch
import pickle
import numpy as np
from torch import nn
from transformers import * 
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader


# Data config
source_dir = './'
suffix = 'rand123'
labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
data_test=pickle.load(open(os.path.join(source_dir, 'data', 'data_test.'+suffix),'rb'))
num_labels = len(labels_ref)

# Model config
max_len = 512
model_checkpoint = os.path.join(source_dir, 'berts', 'bert-base-uncased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, max_len=max_len)#, use_fast=True)
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)).to(device)


def preprocess_function(docu):
    encodings = tokenizer(docu['text'], truncation=True, padding='max_length')   
    repo_name = docu['text'].split(':')[0] 
    input_ids = torch.tensor(encodings['input_ids']).unsqueeze(0)
    att_mask = torch.tensor(encodings['attention_mask']).unsqueeze(0)
    return (input_ids, att_mask, repo_name, docu['labels'])

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

test_dataset = CustomDataset(data_test)  
model_para = sys.argv[1] # First args is the location of the model parameters
model.load_state_dict(torch.load(model_para))
model.to(device)

input_index = sys.argv[2:] # Second args onwards are the test index
for i in input_index:
    assert int(i)<len(test_dataset), f"input index {i} is too large. Input index should be between 0 and {len(test_dataset)}"
    input_ids, input_mask, input_repo, labels = test_dataset[int(i)]
    with torch.no_grad():
        outs = model(input_ids.to(device), attention_mask=input_mask.to(device))
        logit_pred = outs[0]
        pred_label = torch.sigmoid(logit_pred)

        pred_label = pred_label.to('cpu').numpy()
        predictions = np.argsort(pred_label[0])[-5:]
        predictions = predictions[::-1]
        predictions = [p for p in predictions if pred_label[0][p] >= 0.55]

    print(f"Link to repository: https://github.com/{input_repo}")
    print("Predicted labels: "+ str([labels_ref[p] for p in predictions]))
    print("Ground truth labels: "+ str(labels))