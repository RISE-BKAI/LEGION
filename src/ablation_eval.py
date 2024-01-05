import os,sys
import torch
import pickle
import numpy as np
from torch import nn
from transformers import * 
from sklearn.metrics import f1_score, precision_score, recall_score

########## Configuration Part 1 ###########
source_dir = './'
suffix = 'rand123'
model_name = 'bert_base'
model_checkpoint = os.path.join(source_dir, 'berts', 'bert-base-uncased')
    
data_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
data_test=pickle.load(open(os.path.join(source_dir, 'data', 'data_test.'+suffix),'rb'))
labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
num_labels = len(labels_ref)
max_len = 512
batch_size = 32


########## set up ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, max_len=max_len)#, use_fast=True)
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)).to(device)

########## data preprocessing (one-off configuration based on the input data) ###########
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

validation_dataloader = DataLoader(CustomDataset(data_val), shuffle=False, batch_size=batch_size)
test_dataloader = DataLoader(CustomDataset(data_test), shuffle=False, batch_size=batch_size)  

group_head = [i for i, x in enumerate(class_freq) if x>=30]
group_med = [i for i, x in enumerate(class_freq) if x<30 and x>4]
group_tail = [i for i, x in enumerate(class_freq) if x<=4]
print('Label count for head, med and tail groups', len(group_head), len(group_med), len(group_tail))

model_paras = []
model_dir = os.path.join(source_dir, 'models')

for j, fname in enumerate(os.listdir(model_dir)):
    if not fname.startswith(".") and not os.path.isdir(os.path.join(model_dir, fname)) and model_name in fname:
        loss_func_name = fname.split("_")[3]
        if loss_func_name == "DBloss":
            model_paras.append((loss_func_name, os.path.join(model_dir, fname), True))
            model_paras.append((loss_func_name, os.path.join(model_dir, fname), False))
        elif loss_func_name == "BCE":
            model_paras.append((loss_func_name, os.path.join(model_dir, fname), True))
        
res = []*10
for loss_func_name, model_para,threshold_bool in model_paras:
    res_model = []
    model.load_state_dict(torch.load(model_para))
    ########## start training + val ###########
    
    model.eval()

    for top_k in [1,3,5,8]:
        res_k = []
        true_labels,pred_labels = [],[]   
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
            # Forward pass
                outs = model(b_input_ids, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)
        
                pred_label = pred_label.to('cpu').numpy()
                b_labels = b_labels.to('cpu').numpy()
            true_labels.append(b_labels)
            pred_labels.append(pred_label)
        
        true_labels_val = [item for sublist in true_labels for item in sublist]
        pred_labels_val = [item for sublist in pred_labels for item in sublist]

        true_labels,pred_labels = [],[]
        true_labels_all,pred_labels_all = [],[]
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
            # Forward pass
                outs = model(b_input_ids, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)

                pred_label = pred_label.to('cpu').numpy()
                b_labels = b_labels.to('cpu').numpy()

                predictions = [np.argsort(row)[-top_k:] for row in pred_label]
                predictions = [row[::-1] for row in predictions]

                for n in range(len(pred_label)):
                    for t in range(len(pred_label[n])):
                        if t not in predictions[n]:
                            pred_label[n][t] = 0
            true_labels.append(b_labels)
            pred_labels.append(pred_label)
        
        true_labels_test = [item for sublist in true_labels for item in sublist]
        pred_labels_test = [item for sublist in pred_labels for item in sublist]
        
        true_bools = [tl==1 for tl in true_labels_val]
        default_th = 0.5
        thresholds = (np.array(range(-50,51))/100)+default_th 
        if threshold_bool == False:
            thresholds = [0.5]
        f1_results_micro = []
        for th in thresholds:
            pred_bools = [pl>th for pl in np.array(pred_labels_val)]
            f1_results_micro.append(f1_score(true_bools,pred_bools,average='micro', zero_division=0))
        best_micro_f1_th = thresholds[np.argmax(f1_results_micro)]

        miF_val = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_val)],average='micro', zero_division=0)
        
        true_bools = [tl==1 for tl in true_labels_test]
        p_test = precision_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_test)],average='micro', zero_division=0)
        re_test = recall_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_test)],average='micro', zero_division=0)
        miF_test = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_test)],average='micro', zero_division=0)
        res_k.extend([loss_func_name, top_k, best_micro_f1_th, miF_val, p_test, re_test, miF_test])
        
        # evaluation with same cutoff
        for group_name, group in [('Head',group_head), ('Medium', group_med), ('Tail',group_tail)]:
            true_bools = [[tl[i]==1 for i in group] for tl in true_labels_val]
            pred_labels_sub = [[pl[i] for i in group] for pl in pred_labels_val]
            miF_val = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_sub)],average='micro', zero_division=0)
            
            true_bools = [[tl[i]==1 for i in group] for tl in true_labels_test]
            pred_labels_sub = [[pl[i] for i in group] for pl in pred_labels_test]
            miF_test = f1_score(true_bools,[pl>best_micro_f1_th for pl in np.array(pred_labels_sub)],average='micro', zero_division=0)
            res_k.extend([miF_val, miF_test])
        
        res_model.append(res_k)
    print(res_model)
    res += res_model.copy()
    
import pandas as pd
df_res = pd.DataFrame(res)
df_res.columns = ["Loss Function Name", "Top k", "Threshold", 
                  "F1-val-All", "Precision-test-All", "Recall-test-All", "F1-test-All", 
                  "F1-val-Head", "F1-test-Head",
                  "F1-val-Medium", "F1-test-Medium",
                  "F1-val-Tail", "F1-test-Tail"]
df_res.to_excel(f'eval_{model_name}.xlsx',index=False)