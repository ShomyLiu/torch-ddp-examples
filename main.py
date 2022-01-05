# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from transformers import BertForSequenceClassification

from utils import bert_name, collate_fn, load_data_and_labels, Data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time

epochs = 15
lr = 2e-5
batch_size = 64


def train():

    # dist init
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # dataset
    x_text, y = load_data_and_labels("./data/rt-polarity.pos", "./data/rt-polarity.neg")
    x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=0.1)
    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # model
    model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=2)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = -0.1
    print("start training...")
    for epoch in range(1, epochs):
        total_loss = 0.0
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
        print(f"\t Epoch{epoch}: loss: {total_loss:.4f}, acc: {acc:.4f}, time: {(end_time - start_time):.2f}s")
        if acc > best_acc:
            best_acc = acc

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
            preds.append(predict)
            labels.append(truth)

    model.train()
    predict = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    correct = (predict == labels).sum().item()
    return correct * 1.0 / len(predict)


train()
