import sys
import os
import torch
import pickle
import json
import numpy as np
from torch import nn
from transformers import *
from tqdm import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

########## Configuration Part 1 ###########
source_dir = './'
suffix = 'rand123'
loss_func_name = str(sys.argv[1]) # The loss function name will be given as first argument
batch_size = int(sys.argv[2]) # The batch size will be given as second argument
epochs = int(sys.argv[3]) # The epoch will be given as third argument
model_name = str(sys.argv[4]) # model name as the fourth argument

if model_name == 'bert_base':
    model_checkpoint = os.path.join(source_dir, 'berts', 'bert-base-uncased')
elif model_name == 'roberta_base':
    model_checkpoint = os.path.join(source_dir, 'berts', 'roberta-base')
elif model_name == 'bart_base':
    model_checkpoint = os.path.join(source_dir, 'berts', 'bart-base')
elif model_name == 'electra_base':
    model_checkpoint = os.path.join(source_dir, 'berts', 'electra-base-discriminator')

data_train=pickle.load(open(os.path.join(source_dir, 'data', 'data_train.'+suffix),'rb'))
data_val=pickle.load(open(os.path.join(source_dir, 'data', 'data_val.'+suffix),'rb'))
labels_ref=pickle.load(open(os.path.join(source_dir, 'data', 'labels_ref.'+suffix),'rb'))
class_freq=pickle.load(open(os.path.join(source_dir, 'data', 'class_freq.'+suffix),'rb'))
train_num=pickle.load(open(os.path.join(source_dir, 'data', 'train_num.'+suffix),'rb'))
num_labels = len(labels_ref)
max_len = 512

########## set up ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, max_len=max_len)
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels),device_ids=[0]).to(device)

########## Configuration Part 2 ###########

from util_loss import ResampleLoss

if loss_func_name == 'BCE':
    lr = 1e-5
    loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                             focal=dict(focal=False, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             class_freq=class_freq, train_num=train_num)

if loss_func_name == 'FL':
    lr = 1e-5
    loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             class_freq=class_freq, train_num=train_num)
    
if loss_func_name == 'DBloss': # DB
    lr = 1e-4
    loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
                             class_freq=class_freq, train_num=train_num)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight'] 
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,lr=lr) # consider the scale of loss function
        
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
    
########## start training + val ###########
torch.manual_seed(123)
train_dataloader = DataLoader(CustomDataset(data_train), shuffle=False, batch_size=batch_size)
validation_dataloader = DataLoader(CustomDataset(data_val), shuffle=False, batch_size=batch_size)

best_f1_for_epoch = 0
train_loss_over_time = []
eval_loss_over_time = []

for epoch in tqdm(range(epochs), desc="Epoch"):
    print(f"Epoch: {epoch}")
    # Training
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
  
    for bi, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()

        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs[0]
        loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels))
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    train_loss_over_time.append(tr_loss/nb_tr_steps)

    # Validation
    model.eval()
    val_loss = 0
    nb_val_steps = 0
    true_labels,pred_labels = [],[]
    
    for _, batch in enumerate(validation_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)
            loss = loss_func(b_logit_pred.view(-1,num_labels),b_labels.type_as(b_logit_pred).view(-1,num_labels))
            val_loss += loss.item()
            nb_val_steps += 1
    
            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    print("Validation loss: {}".format(val_loss/nb_val_steps))
    eval_loss_over_time.append(val_loss/nb_val_steps)

    # Flatten outputs
    true_labels = [item for sublist in true_labels for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]

    # Calculate Accuracy
    threshold = 0.5
    true_bools = [tl==1 for tl in true_labels]
    pred_bools = [pl>threshold for pl in pred_labels]
    print("Size of true bools, predict bools: " + str(np.shape(true_bools)) + ", " + str(np.shape(pred_bools)))
    val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro',zero_division=0)
    val_precision_accuracy = precision_score(true_bools, pred_bools,average='micro',zero_division=0)
    val_recall_accuracy = recall_score(true_bools, pred_bools,average='micro',zero_division=0)
    
    print('F1 Validation Default: ', val_f1_accuracy)
    print('Precision Validation Default: ', val_precision_accuracy)
    print('Recall Validation Default: ', val_recall_accuracy)

    # Calculate AUC as well
    val_auc_score = roc_auc_score(true_bools, pred_labels, average='micro')
    print('AUC Validation: ', val_auc_score)
    
    # Search best threshold for F1
    best_med_th = 0.5
    micro_thresholds = (np.array(range(-10,11))/100)+best_med_th
    f1_results, prec_results, recall_results = [], [], []
    for th in micro_thresholds:
        pred_bools = [pl>th for pl in pred_labels]
        test_f1_accuracy = f1_score(true_bools,pred_bools,average='micro',zero_division=0)
        test_precision_accuracy = precision_score(true_bools, pred_bools,average='micro',zero_division=0)
        test_recall_accuracy = recall_score(true_bools, pred_bools,average='micro',zero_division=0)
        f1_results.append(test_f1_accuracy)
        prec_results.append(test_precision_accuracy)
        recall_results.append(test_recall_accuracy)

    best_f1_idx = np.argmax(f1_results) #best threshold value

    # Print report
    print('Best Threshold: ', micro_thresholds[best_f1_idx])
    print('Best F1 with Threshold: ', f1_results[best_f1_idx])
    
    # Save the model if this epoch gives the best f1 score in validation set
    if f1_results[best_f1_idx] > (best_f1_for_epoch * 0.995):
        best_f1_for_epoch = f1_results[best_f1_idx]
        model_dir = os.path.join(source_dir, 'models')
        for fname in os.listdir(model_dir):
            if fname.startswith('_'.join([model_name,loss_func_name,suffix])):
                os.remove(os.path.join(model_dir, fname))
        torch.save(model.state_dict(), os.path.join(model_dir, '_'.join([model_name,loss_func_name,suffix,'epoch'])+str(epoch+1)+'para'))

    
    log_dir = os.path.join(source_dir, 'logs')
    # Log all results in validation set with different thresholds
    with open(os.path.join(log_dir, '_'.join([model_name,loss_func_name,suffix,'epoch'])+str(epoch+1)+'.json'),'w') as f:
        d = {}
        d["f1_accuracy_default"] =  val_f1_accuracy
        d["pr_accuracy_default"] =  val_precision_accuracy
        d["rec_accuracy_default"] =  val_recall_accuracy
        d["auc_score_default"] =  val_auc_score
        d["thresholds"] =  list(micro_thresholds)
        d["threshold_f1s"] =  f1_results
        d["threshold_precisions"] =  prec_results
        d["threshold_recalls"] =  recall_results
        json.dump(d, f)
    
    open(os.path.join(log_dir, '_'.join([model_name,loss_func_name,suffix,'epoch'])+str(epoch+1)+'.tmp'),'w').write('%s %s' % (micro_thresholds[best_f1_idx], f1_results[best_f1_idx]))

epochs_trained = len(train_loss_over_time)
if epochs_trained != epochs:
    print(f"Training was interrupted at epoch {epochs_trained}.")

plt.plot(range(1, epochs_trained+1), train_loss_over_time, label="Train loss")
plt.plot(range(1, epochs_trained+1), eval_loss_over_time, label="Eval loss")
plt.title(f"{loss_func_name} during training")
plt.legend(loc="upper left")
plt.savefig(os.path.join(source_dir,'logs',f'{model_name}_{loss_func_name}_over_{epochs}_epochs.png'))