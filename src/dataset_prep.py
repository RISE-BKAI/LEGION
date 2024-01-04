#!/usr/bin/env python
# coding: utf-8

# ## Before training
import json
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle


data_train_all = json.loads(open('data/training_data.json').read())
data_test = json.loads(open('data/test_data.json').read())
print(len(data_train_all)+len(data_test))

data_train, data_val = train_test_split(data_train_all, random_state=123, test_size=1000)
print(len(data_train), len(data_val), len(data_test))

term2count = Counter([x for docu in data_train for x in docu['labels']])
FREQ_CUTOFF = 0 
term_freq = sorted([term for term, count in term2count.items() if count>=FREQ_CUTOFF])
labels_ref = sorted([z for z in set([y for x in data_train for y in x['labels']]) if z in term_freq]) 
print(len(term2count), len(labels_ref))
class_freq = [term2count[x] for x in labels_ref]
train_num = len(data_train)

pickle.dump(data_train, open('data/data_train.rand123','wb')) 
pickle.dump(data_val, open('data/data_val.rand123','wb')) 
pickle.dump(data_test, open('data/data_test.rand123','wb')) 
pickle.dump(labels_ref, open('data/labels_ref.rand123','wb')) 
pickle.dump(class_freq, open('data/class_freq.rand123','wb')) 
pickle.dump(train_num, open('data/train_num.rand123','wb')) 


# #### to choose the proper gamma of map_param in dbloss

r_all = []
for docu in data_train:
    docu_p = [1/term2count[x] for x in docu['labels']]
    docu_p_sum = sum(docu_p)
    r_all.extend([p/docu_p_sum for p in docu_p])
    
print(np.mean(r_all),np.median(r_all))
print((np.mean(r_all) + np.median(r_all)) / 2)
## thus 0.9 can be a good one intuitively





