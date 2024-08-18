import warnings
import joblib
from imblearn.over_sampling import SMOTE
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('./train/train_no_0.csv',low_memory=False)  # 示例：从CSV文件中读取数据


def compute_pearson_similarity1(features):
    n = features.shape[0]  # 样本数量
    similarity_matrix = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity = 1
            else:
                similarity, _ = pearsonr(features.iloc[i, :], features.iloc[j, :])
                similarity = (similarity + 1) / 2
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    return similarity_matrix

def find_neighbors(similarity_matrix, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='precomputed')
    nbrs.fit(1 - similarity_matrix)
    distances, indices = nbrs.kneighbors(1 - similarity_matrix)
    return distances, indices
def compute_pearson_similarity(features):
    if np.any(np.std(features, axis=0) == 0):
        warnings.warn("One or more features have zero variance.")
    correlation_matrix = np.corrcoef(features.T, rowvar=False)
    np.nan_to_num(correlation_matrix, copy=False)
    similarity_matrix = (correlation_matrix + 1) / 2
    return similarity_matrix


def compute_fixed_k_neighbors_with_threshold(features, similarity_threshold=0.8, k=30):
    similarity_matrix = compute_pearson_similarity(features)
    nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed')
    nbrs.fit(1-similarity_matrix)
    distances, indices = nbrs.kneighbors(1-similarity_matrix)

    smoothed_similarities = []
    dynamic_indices = []

    for i in range(similarity_matrix.shape[0]):

        neighbor_similarities = similarity_matrix[i, indices[i]]

        valid_mask = neighbor_similarities > similarity_threshold
        valid_indices = indices[i][valid_mask]
        valid_similarities = neighbor_similarities[valid_mask]

        dynamic_indices.append(valid_indices)

        epsilon = 1e-5
        smoothed_sim = 1 / (1 - valid_similarities + epsilon)
        smoothed_similarities.append(smoothed_sim)
    return smoothed_similarities, dynamic_indices, [len(di) for di in dynamic_indices]


def create_single_sample_graph(df, sample_index, knn_indices, knn_weights,pbar):
    import networkx as nx
    import numpy as np
    G = nx.Graph()
    label = df.iloc[sample_index, -1]
    all_features = df.iloc[:,:-1].values
    sample_scaled_features = all_features[sample_index]
    G.add_node(sample_index, feature=sample_scaled_features)

    for i, neighbor_index in enumerate(knn_indices[sample_index]):
        neighbor_features = all_features[neighbor_index]
        G.add_node(neighbor_index, features=neighbor_features)
        weight = knn_weights[sample_index][i]
        G.add_edge(sample_index, neighbor_index, weight=weight)

    original_node_indices = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(original_node_indices)}

    G = nx.relabel_nodes(G, mapping=node_mapping)

    node_features_list = [data['features'] for _, data in G.nodes(data=True)]
    node_features_array = np.array(node_features_list, dtype=np.float32)
    node_features_tensor = torch.from_numpy(node_features_array).to(device)

    edge_index = torch.tensor([[s, t] for s, t in G.edges()], dtype=torch.long).t().contiguous().to(device)
    edge_weight = torch.tensor([G[s][t]['weight'] for s, t in G.edges()], dtype=torch.float).to(device)

    y = torch.tensor([label], dtype=torch.long).to(device)
    train_mask = torch.zeros(len(G), dtype=torch.bool).to(device)
    train_mask[0] = True

    data = Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_weight, y=y, train_mask=train_mask)
    pbar.update(1)
    return data
# 使用KNN选择邻居
cols = ['Chr', 'Start', 'End', 'Ref', 'Alt', 'Func.refGene', 'Gene.refGene', 'GeneDetail.refGene',
        'AAChange.refGene',  'MetaSVM_pred', 'MetaLR_pred', 'M-CAP_pred', 'DANN_pred', 'MutPred_pred',
        'MVP_pred', 'REVEL_pred',
        'CADD_pred', 'MutationTaster_pred', 'VEST4_pred',
        'MetaSVM_score', 'MetaLR_score', 'M-CAP_score', 'DANN_rankscore', 'MutPred_score', 'MVP_score',
        'CADD_phred', 'MutationTaster_score', 'VEST4_score', 'REVEL_score']
col1 = ['CLNSIG']

col_to_keep = df[cols].copy()
X = df.drop(cols+col1, axis=1)
y = df['CLNSIG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
col_to_keep_test = col_to_keep.iloc[X_test.index]

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

df_resampled = pd.DataFrame(X_res, columns=X_train.columns)
df1_resampled = pd.DataFrame(X_test, columns=X_test.columns)
target_df = pd.DataFrame(y_res, columns=['CLNSIG'])
target1_df = pd.DataFrame(y_test, columns=['CLNSIG'])

df_resampled = pd.concat([df_resampled, target_df], axis=1)
df1_resampled = pd.concat([df1_resampled, target1_df], axis=1)

print(df['CLNSIG'].value_counts())
print(df_resampled['CLNSIG'].value_counts())
print(df1_resampled['CLNSIG'].value_counts())
df = df_resampled.copy()
df1 = df1_resampled.copy()
scaler = MinMaxScaler(feature_range=(0, 1))
col = ['func','Chr','Start','End','Ref','Alt','Func.refGene','Gene.refGene','GeneDetail.refGene','AAChange.refGene','CLNSIG',
       'MetaSVM_pred', 'MetaLR_pred', 'M-CAP_pred', 'DANN_pred', 'MutPred_pred', 'MVP_pred', 'REVEL_pred','CADD_phred',
       'MutationTaster_pred', 'VEST4_pred','MetaSVM_score', 'MetaLR_score', 'M-CAP_score',
       'DANN_rankscore', 'MutPred_score', 'MVP_score','CADD_pred', 'MutationTaster_score', 'VEST4_score', 'REVEL_score',
       'blosum100','Gm12878', 'H1hesc', 'Hepg2','Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
        'hydrophilic', 'hydrophobic', 'amphipathic ', 'cyclic',
        'essential', 'aromatic', 'aliphatic', 'nonpolar', 'polar_uncharged', 'acidic','basic',
       'sulfur'
       ]
col2 = ['AF', 'AF_raw','AF_male', 'AF_female', 'AF_afr', 'AF_ami', 'AF_amr',
        'AF_asj', 'AF_eas', 'AF_fin', 'AF_nfe', 'AF_oth', 'AF_sas',
        'gdi', 'gdi_phred', 'rvis1', 'rvis2', 'lof_score',
        'DS_AG', 'DS_AL', 'DS_DG','DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',
        'GERP++_NR','GERP++_RS','phyloP100way_vertebrate','phyloP30way_mammalian','phyloP17way_primate',
        'phastCons100way_vertebrate','phastCons30way_mammalian','phastCons17way_primate',
        'GERP++_RS_rankscore','phyloP100way_vertebrate_rankscore','phyloP20way_mammalian',
        'phyloP20way_mammalian_rankscore','phastCons100way_vertebrate_rankscore',
        'phastCons20way_mammalian','phastCons20way_mammalian_rankscore','SiPhy_29way_logOdds',
        'SiPhy_29way_logOdds_rankscore',
        'molecular_weight',
        'equipotential_point','pka_cooh', 'pka_nh3','0','1','2','3','4','5','6','7','8','9'
       ]

df.columns = df.columns.astype(str)
df1.columns =df1.columns.astype(str)
df[col2] = scaler.fit_transform(df[col2])
df1[col2] = scaler.transform(df1[col2])

joblib.dump(scaler, 'scaler.gz')
print("ok")

print("=============================================")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def create(df):
    data_list = []
    feature1 = df.iloc[:, :13]
    feature2 = df.iloc[:, 19:61]
    feature3 = df.iloc[:, 61:70]
    feature1 = pd.concat([feature1, feature2], axis=1)
    features = pd.concat([feature1, feature3], axis=1)
    dynamic_distances, dynamic_indices, k_values = compute_fixed_k_neighbors_with_threshold(features)
    print(k_values)
    len1 = len(df)

    pbar = tqdm(total=len1)
    epoch_start_time = time.time()
    for sample_index in range(len1):
        data = create_single_sample_graph(df, sample_index, dynamic_indices, dynamic_distances,pbar)
        data_list.append(data)
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    pbar.close()
    return data_list

data1 = create(df)
data2 = create(df1)
X_test = pd.concat([col_to_keep_test,df1], axis=1)
X_test.to_csv('./train/val.csv',index=False)
filename1 = 'train.pth'
with open(filename1, 'wb') as file:
    torch.save(data1, file)
print(f"数据已保存到文件：{filename1}")
filename2 = 'val.pth'
with open(filename2, 'wb') as file:
    torch.save(data2, file)
print(f"数据已保存到文件：{filename2}")