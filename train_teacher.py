import torch
import numpy as np
import math
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification,AutoTokenizer
from load_data import prep_id, prep_sym1, prep_sym2
from tokenize_data import get_sample, tokenize_batch
from utilis import flat_accuracy


def bias_train(batch_size):
    bias_model.train()
    bias_train_loss = 0
    iter_num = 0
    total_iter = len(train_dataloader)
    
    for batch_data in train_dataloader:
        
        bias_optim.zero_grad()
        input_ids,attention_masks,labels = tokenize_batch(batch_data,batch_size)
        
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        
        outputs = bias_model(input_ids, attention_mask=attention_masks, labels=labels)
        
        loss = outputs[0]
        bias_train_loss += loss.item()

        loss.backward()
        
        bias_optim.step()
        bias_scheduler.step()

        iter_num += 1
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
        
    print("Epoch: %d, Average training loss for bias model: %.4f"%(epoch, bias_train_loss/len(train_dataloader)))

    
def validation(model,dataloader,batch_size):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    
    for batch_data in dataloader:
        with torch.no_grad():
            
            input_ids,attention_masks,labels = tokenize_batch(batch_data,batch_size)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        
        loss = outputs[0]
        logits = outputs[1]
    
        total_eval_loss += loss.item()
        logits = logits.detach().to('cpu').numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f"%(total_eval_loss/len(dataloader)))
    print("-------------------------------")

torch.manual_seed(999)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bias_model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=3)
device = torch.device("cuda")
bias_model.to(device)
bias_optim = torch.optim.Adam(bias_model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

train_s1, train_s2, train_label = prep_id('dataset/fever_train.jsonl')
dev_s1, dev_s2, dev_label = prep_id('dataset/fever_dev.jsonl')

symv1_s1,symv1_s2,symv1_label = prep_sym1('dataset/symv1.txt')
symv2_s1,symv2_s2,symv2_label = prep_sym2('dataset/symv2.txt')

train_dataloader = get_sample(train_s1,train_s2,train_label,16)
valid_dataloader = get_sample(dev_s1,dev_s2,dev_label,16)
sym1_dataloader = get_sample(symv1_s1,symv1_s2,symv1_label,16)
sym2_dataloader = get_sample(symv2_s1,symv2_s2,symv2_label,16)
total_steps = len(train_dataloader) * 5
bias_scheduler = get_linear_schedule_with_warmup(bias_optim, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

import os
from datetime import datetime
output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())

for epoch in range(5):
    print("------------Epoch: %d ----------------" % epoch)
    bias_train(16)
    validation(bias_model,valid_dataloader,16)
    print('Now evaluating on sym1 set.')
    validation(bias_model,sym1_dataloader,16)
    print("-------------------------------")
    print('Now evaluating on sym2 set.')
    validation(bias_model,sym2_dataloader,16)
    print("-------------------------------")
    output_path1 = output_dir + "bias.weights" + str(epoch)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    torch.save(bias_model.state_dict(), output_path1)
    print('model saved')
print('finished')