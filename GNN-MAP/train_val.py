from collections import Counter
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import time
from torch_geometric.loader import DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, f1_score, recall_score, \
    precision_score, precision_recall_curve, roc_auc_score
from train.model import GraphClassifier
from train.tool import initialize_table, update_table, MyDataset, custom_collate_fn

filename = '../train.pth'
with open('../train.pth', 'rb') as file:
    train_list = torch.load(file)
with open('../val.pth', 'rb') as file:
    val_list = torch.load(file)

# 训练函数
def train_model(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, return_proba=False)
        loss = criterion(out[:,0], data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.update(1)
    return total_loss / len(loader)

def evaluate_model(model, loader, device):

    model.eval()
    preds, labels = [], []
    total_loss = 0
    y_pred_prob = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, return_proba=False)
            loss = criterion(out[:,0], data.y.float())
            prob = model.predict_proba(data.x,data.edge_index)
            pred = (prob > 0.5).float()
            total_loss += loss.item()
            preds.append(pred.cpu())
            labels.append(data.y.cpu())
            y_pred_prob.append(prob.cpu())
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    y_pred_prob = torch.cat(y_pred_prob, dim=0)
    precision_curve, recall_curve, _ = precision_recall_curve(labels.numpy(),y_pred_prob.numpy())
    auprc = auc(recall_curve, precision_curve)
    auc_score = roc_auc_score(labels.numpy(), y_pred_prob.numpy())
    return total_loss / len(loader),\
        accuracy_score(labels.numpy(), preds.numpy()), \
           precision_score(labels.numpy(), preds.numpy()), \
           recall_score(labels.numpy(), preds.numpy()), \
           f1_score(labels.numpy(), preds.numpy()),auprc,auc_score



num_epochs = 100
results = []
x_features = [data.x for data in train_list]
print(x_features[0].size(1))
num_features = x_features[0].size(1)
num_classes = 1

table = initialize_table()
print("训练集类别分布:", Counter([data.y.item() for data in train_list]))
print("测试集类别分布:", Counter([data.y.item() for data in val_list]))
train_loader = DataLoader(train_list, batch_size=1, shuffle=True)
val_loader = DataLoader(val_list, batch_size=1, shuffle=False)
model = GraphClassifier(num_features, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0012)
pos_weight = torch.tensor([7818 / 1087]).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
scheduler = ExponentialLR(optimizer, gamma=0.85)
patience = 5
best_val_loss = float('inf')
early_stopping_counter = 0
epoch_result = []
for epoch in range(100):
    status = ""
    epoch_start_time = time.time()
    current_time = datetime.fromtimestamp(epoch_start_time)
    print(current_time)
    time.sleep(2)
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/100", ncols=100)
    train_loss = train_model(model, train_loader, optimizer, device)
    pbar.close()
    loss,acc, prec, rec, f1, auprc, auc_score = evaluate_model(model, val_loader, device)
    scheduler.step()

    if min(best_val_loss, loss * 1.05) >= loss:
        best_val_loss = loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), './result/trained22_model.pth')
        status = "Model saved!"
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            break
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    epoch_result.append((acc, prec, rec, f1))
    update_table(table,f"{1}/{5}",f"{epoch+1}/{100}", train_loss, loss, acc*100, prec*100, rec*100, f1*100,auprc*100,auc_score*100, status)
    if (epoch+1) % 5 == 0:
        print(table)
        table.clear_rows()
    else:
        print(
            f'Epoch {epoch + 1}, train_loss: {train_loss:.4f},val_loss:{loss:.4f} Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f},AUC:{auc_score:.4f},AUPRC:{auprc:.4f}')