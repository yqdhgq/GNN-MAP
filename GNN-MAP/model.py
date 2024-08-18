import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, TopKPooling


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(3, 1))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, gcn_out, gat_out, sage_out):
        # Stack outputs
        stacked_outputs = torch.stack([gcn_out, gat_out, sage_out], dim=0)
        # Compute attention weights
        attention_scores = self.softmax(self.attention_weights)
        # 扩展权重以匹配输出的维度 [3, 1, 1] -> [3, N, F]
        attention_scores = attention_scores.view(3, 1, 1)
        # Apply attention weights
        weighted_output = torch.sum(attention_scores * stacked_outputs, dim=0)
        return weighted_output

class HybridGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout_rate=0.2):
        super(HybridGraphConv, self).__init__()
        self.dropout_rate = dropout_rate
        self.gcn = GCNConv(in_channels, out_channels)
        self.gat = GATConv(in_channels, out_channels, heads=heads,dropout=dropout_rate)
        self.gat_transform = nn.Linear(out_channels * heads, out_channels)  # 将GAT输出映射到统一维度
        self.sage = SAGEConv(in_channels, out_channels)
        self.attention = AttentionModule(out_channels)
        self.batch_norm = LayerNorm(out_channels)
        self.fusion = nn.Linear(out_channels * 2, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = None

    def forward(self, x, edge_index):
        identity = x if self.residual is None else self.residual(x)
        gcn_x = F.relu(self.gcn(x, edge_index))
        gat_x = F.relu(self.gat(x, edge_index))
        gat_x = self.gat_transform(gat_x)
        sage_x = F.relu(self.sage(x, edge_index))
        merged_x = self.attention(gcn_x, gat_x, sage_x)
        merged_x = self.batch_norm(merged_x)
        merged_x = self.dropout(merged_x)
        out = self.fusion(torch.cat([merged_x, identity], dim=1))
        return F.relu(out + identity)

class AdaptiveFeaturePooling(nn.Module):
    def __init__(self, in_channels,out_channels, dropout_rate=0.5):
        super(AdaptiveFeaturePooling, self).__init__()
        self.dropout_rate = dropout_rate
        self.attn = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x,edge_index, batch=None):
        weights = torch.sigmoid(self.attn(x))
        weighted_x = weights * x
        weighted_x = self.dropout(weighted_x)
        mean = global_mean_pool(weighted_x, batch=batch)
        max = global_max_pool(x, batch=batch)
        return torch.cat([max,mean],dim=1)

class GraphClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphClassifier, self).__init__()
        self.conv1 = HybridGraphConv(num_features, 128, dropout_rate=0.3)
        self.conv2 = HybridGraphConv(128, 64, dropout_rate=0.4)
        self.conv3 = HybridGraphConv(64, 32, dropout_rate=0.5)
        self.pool = AdaptiveFeaturePooling(32, 64,dropout_rate=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.linear = nn.Linear(32, num_classes)

    def forward(self, x, edge_index, return_proba=False):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.pool(x,edge_index)
        x = self.classifier(x)
        logits = self.linear(x)
        if return_proba:
            return torch.sigmoid(logits)
        return logits

    def predict_proba(self, x, edge_index):
        # 确保模型在评估模式
        self.eval()
        # 不需要计算梯度
        with torch.no_grad():
            return self.forward(x, edge_index, return_proba=True)

