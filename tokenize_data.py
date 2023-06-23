
import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset,DataLoader, Dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class CustomDataset(Dataset):
    def __init__(self, sent1, sent2, labels):
        self.sent1 = sent1
        self.sent2 = sent2
        self.labels = labels
        
    def __len__(self):
        return len(self.sent1)
    
    def __getitem__(self, index):
        sent1 = self.sent1[index]
        sent2 = self.sent2[index]
        label = self.labels[index]
        
        return sent1, sent2, label

def get_sample(sent1,sent2,label,size):
    train_dataset= CustomDataset(sent1,sent2,label)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=size, shuffle=True)
    return train_dataloader

def tokenize_batch(batch,batch_size):
    
    sent1 = batch[0]
    sent2 = batch[1]
    max_len = 0
    real_size = len(sent1)
    if real_size < batch_size:
        batch_size = real_size
    
    for i in range(batch_size):
        input_ids = tokenizer.encode(sent1[i],sent2[i],truncation=True,padding = 'longest',add_special_tokens=True)
        max_len = max(max_len,len(input_ids))
    
    input_ids = []
    attention_masks = []
            
    for i in range(batch_size):
        encoded_dict = tokenizer.encode_plus(
                        sent1[i],  
                        sent2[i],                    
                        add_special_tokens = True,
                        truncation=True,          
                        max_length = max_len,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )
      
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(batch[2])
    
    return input_ids,attention_masks,labels
