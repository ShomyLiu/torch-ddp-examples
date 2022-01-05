# -*- coding: utf-8 -*-

import os
import torch
import torch.optim as optim
import torch.distributed as dist
from transformers import BertForSequenceClassification
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd

from utils import bert_name, collate_fn, load_data_and_labels, Data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time

epochs = 15
lr = 2e-5
batch_size = 64

hvd.init()


def train():

    # dist init
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # dataset
    x_text, y = load_data_and_labels("./data/rt-polarity.pos", "./data/rt-polarity.neg")
    x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=0.1)
    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)
    train_sampler = DistributedSampler(train_data, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(train_data, batch_size=batch_size*size, collate_fn=collate_fn, sampler=train_sampler)
    test_sampler = DistributedSampler(test_data, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = DataLoader(test_data, batch_size=batch_size*size, shuffle=False, collate_fn=collate_fn, sampler=test_sampler)

    # model
    model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=2)
    model.to(device)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    optimizer = optim.Adam(model.parameters(), lr=lr*size)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    best_acc = -0.1
    print(f"rank {rank} start training...")
    for epoch in range(1, epochs):
        total_loss = 0.0
        train_sampler.set_epoch(epoch)
        model.train()
        start_time = time.time()
        for step, batch_data in enumerate(train_loader):
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(**inputs, labels=labels)
            loss = output[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        end_time = time.time()
        acc = test(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            if rank == 0:
                print(f"\t Epoch{epoch}: loss: {total_loss:.4f}, acc: {acc:.4f}, time: {(end_time - start_time):.2f}s")

    if rank == 0:
        print("*"*20)
        print(f"finished; best acc: {best_acc:.4f}")


def test(model, test_loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for data in test_loader:
            inputs, truth = data
            inputs = inputs.to(device)
            truth = truth.to(device)
            output = model(**inputs)['logits']
            predict = torch.max(output.data, 1)[1]

            cur_preds = hvd.allgather(predict)
            cur_truth = hvd.allgather(truth)

            preds.append(cur_preds)
            labels.append(cur_truth)

    model.train()
    predict = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    correct = (predict == labels).sum().item()
    return correct * 1.0 / len(predict)


train()
