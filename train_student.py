import torch
import numpy as np
import math
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification,AutoTokenizer
from load_data import prep_id, prep_sym1, prep_sym2
from tokenize_data import get_sample, tokenize_batch
from utilis import flat_accuracy, smooth_label


def debias_train(batch_size,method):
    debias_model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_dataloader)
    loss_func = torch.nn.CrossEntropyLoss()
    
    for batch_data in train_dataloader:
       
        debias_optim.zero_grad()
        input_ids,attention_masks,labels = tokenize_batch(batch_data,batch_size)
        y_true3 = torch.nn.functional.one_hot(labels, num_classes=3)
        y_true3 = y_true3.to(device)
        
        bias_outputs = bias_model(input_ids=input_ids.to(device), attention_mask=attention_masks.to(device), labels=labels.to(device))
        with torch.cuda.amp.autocast(): #1
            debias_outputs = debias_model(input_ids=input_ids.to(device), attention_mask=attention_masks.to(device),labels=labels.to(device))
            bias_labels = torch.nn.functional.softmax(bias_outputs[1])
            smoothed_label = smooth_label(y_true3.float(),bias_labels)

            debias_logits = debias_outputs[1]
            if method == 1:
                loss = loss_func(debias_logits, smoothed_label)
            else:
                loss = debias_outputs[0]
            
        total_train_loss += loss.item()
        scaler.scale(loss).backward()
    
        scaler.step(debias_optim)
        scaler.update()
        debias_scheduler.step()

        iter_num += 1
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
    print("Epoch: %d, Average training loss for debiased model: %.4f"%(epoch, total_train_loss/len(train_dataloader)))
    
def validation(model,dataloader,batch_size,method):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    
    iter_num = 0
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
        if method == 0:
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        else:
            logits = logits[:, :-1]
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        iter_num = iter_num + 1
        if(iter_num % 100==0):
            print("iter_num:",iter_num)
        
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    test_loss = total_eval_loss/len(dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f"%(total_eval_loss/len(dataloader)))
    print("-------------------------------")
    return avg_val_accuracy, test_loss

torch.manual_seed(999)
bias_model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=3)
bias_path = "bias.weights4"
bias_model.load_state_dict(torch.load(bias_path))

debias_model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=4)
device = torch.device("cuda")
bias_model.to(device)
debias_model.to(device)


train_s1, train_s2, train_label = prep_id('dataset/fever_train.jsonl')
dev_s1, dev_s2, dev_label = prep_id('dataset/fever_dev.jsonl')

symv1_s1,symv1_s2,symv1_label = prep_sym1('dataset/symv1.txt')
symv2_s1,symv2_s2,symv2_label = prep_sym2('dataset/symv2.txt')

train_dataloader = get_sample(train_s1,train_s2,train_label,16)
valid_dataloader = get_sample(dev_s1,dev_s2,dev_label,16)
sym1_dataloader = get_sample(symv1_s1,symv1_s2,symv1_label,16)
sym2_dataloader = get_sample(symv2_s1,symv2_s2,symv2_label,16)

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
total_steps = len(train_dataloader) * 5
debias_optim = torch.optim.Adam(debias_model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
debias_scheduler = get_linear_schedule_with_warmup(debias_optim, 
                                            num_warmup_steps = 2000, 
                                            num_training_steps = total_steps)  

for epoch in range(5):
    print("------------Epoch: %d ----------------" % epoch)
    if epoch < 2:
        debias_train(16,0)
    else:
        debias_train(16,1)
    print("-------------------------------")
    print('Now evaluating on validation set.')
    val_acc, val_loss = validation(debias_model,valid_dataloader,16,1)
    print("-------------------------------")
    print('Now evaluating on sym1 set.')
    sym1_acc, sym1_loss = validation(debias_model,sym1_dataloader,16,1)
    print("-------------------------------")
    print('Now evaluating on sym2 set.')
    sym2_acc, sym2_loss = validation(debias_model,sym2_dataloader,16,1)
    print("-------------------------------")

print('finished')
