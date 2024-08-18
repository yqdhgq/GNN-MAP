# 初始化表格并设置列名
from prettytable import PrettyTable
import torch
from torch_geometric.data import DataLoader, Dataset, Data

class MyDataset(Dataset):
    def __init__(self,data_list):
        super(MyDataset, self).__init__()
        # 假设我们有5个图，每个图有不同数量的节点和边
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def custom_collate_fn(batch):
    # 这个函数接收一个列表，列表中的元素是 Dataset 的 __getitem__ 方法返回的结果。
    # 在这里，我们简单地将这个列表直接返回，不进行任何合并操作。
    return batch
def initialize_table():
    table = PrettyTable()
    table.field_names = ["Flod","Epoch", "Loss", "Validation Loss", "Accuracy", "Precision", "Recall", "F1 Score", "AUPRC", "AUC", "Status"]
    return table
# 更新表格信息
def update_table(table, flod,epoch, loss, val_loss, acc, prec, rec, f1,auprc,auc, status):
    table.add_row([flod,epoch, f"{loss:.4f}", f"{val_loss:.4f}", f"{acc:.2f}%", f"{prec:.2f}%", f"{rec:.2f}%", f"{f1:.2f}%",f"{auprc:.2f}",f"{auc:.2f}", status])