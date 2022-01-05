# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from transformers import BertForSequenceClassification
import accelerate

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import collate_fn, bert_name, load_data_and_labels, Data
import time

epochs = 10
lr = 2e-5
batch_size = 64
accelerator = accelerate.Accelerator()


def train():

    x_text, y = load_data_and_labels("./data/rt-polarity.pos", "./data/rt-polarity.neg")
    x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=0.1)

    size = accelerator.num_processes

    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size*size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size*size, shuffle=False, collate_fn=collate_fn)

    accelerator.print(f"train dataset: {len(train_loader.dataset)}, test dataset: {len(test_loader.dataset)}")
    accelerate.utils.set_seed(1234)
    model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=2)

    optimizer = optim.Adam(model.parameters(), lr=lr*size)
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    best_acc = -0.1
    print(f"rank {accelerator.process_index} start training...")
    for epoch in range(1, epochs):
        total_loss = 0.0
        model.train()
        start_time = time.time()
        for step, batch_data in enumerate(train_loader):
            inputs, labels = batch_data
            optimizer.zero_grad()
            output = model(**inputs, labels=labels)
            accelerator.backward(output[0])
            optimizer.step()
            total_loss += output[0].item()
        end_time = time.time()
        acc = test(model, test_loader)
        if acc > best_acc:
            best_acc = acc
        accelerator.print(f"Epoch{epoch}: loss: {total_loss:.4f}, acc: {acc:.4f}, time: {(end_time - start_time):.2f}s")

    accelerator.print("*"*20)
    accelerator.print(f"finished; best acc: {best_acc:.4f}")


def test(model, test_loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for data in test_loader:
            inputs, truth = data
            output = model(**inputs)['logits']
            predict = torch.max(output.data, 1)[1]
            preds.append(accelerator.gather(predict))
            labels.append(accelerator.gather(truth))

    model.train()
    predict = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)

    correct = (predict == labels).sum().item()
    return correct * 1.0 / len(predict)


train()
