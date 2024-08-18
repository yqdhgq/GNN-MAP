import json
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def process_csv(data):
    # 定义要删除的列
    columns_to_drop = [
        'BayesDel_addAF_pred', 'BayesDel_noAF_pred',
        'CADD_raw', 'fathmm-MKL_coding_pred',
        'fathmm-XF_coding_pred', 'Eigen-raw_coding', 'Eigen-phred_coding',
        'Eigen-PC-raw_coding', 'Eigen-PC-phred_coding', 'GenoCanyon_score',
        'LINSIGHT','Interpro_domain.1','GTEx_V6_gene','GTEx_V6_tissue','phastCons100way_vertebrate.1',
        'phyloP100way_vertebrate.1','GERP++_RS.1','SIFT_converted_rankscore','SIFT_pred.1','Polyphen2_HDIV_score','Polyphen2_HDIV_rankscore','Polyphen2_HDIV_pred.1',
        'Polyphen2_HVAR_score','Polyphen2_HVAR_rankscore','Polyphen2_HVAR_pred.1','LRT_score','LRT_converted_rankscore','LRT_pred.1',
        'MutationTaster_converted_rankscore','MutationTaster_pred.1',
        'MutationAssessor_pred.1','FATHMM_pred.1',
        'PROVEAN_pred.1','VEST3_score','VEST3_rankscore','MetaSVM_rankscore','MetaSVM_pred.1',
        'MetaLR_rankscore','MetaLR_pred.1','M-CAP_rankscore','M-CAP_pred.1','CADD_raw.1','CADD_raw_rankscore',
        'CADD_phred.1',	'DANN_score.1','fathmm-MKL_coding_score','fathmm-MKL_coding_rankscore','fathmm-MKL_coding_pred.1',
        'Eigen_coding_or_noncoding','Eigen-raw','Eigen-PC-raw','GenoCanyon_score.1','GenoCanyon_score_rankscore','integrated_fitCons_score.1',
        'integrated_fitCons_score_rankscore','integrated_confidence_value','DamagePredCount','Polyphen2_HDIV_pred',
        'Polyphen2_HVAR_pred','used_aachange','used_nm',
        'GTEx_V8_tissue','Interpro_domain','bStatistic','dbscSNV_ADA_SCORE','dbscSNV_RF_SCORE',
        'GM12878_fitCons_score', 'H1 - hESC_fitCons_score', 'HUVEC_fitCons_score',
        'H1-hESC_fitCons_score', 'non_topmed_AF_popmax', 'non_cancer_AF_popmax','MPC_score'
        'non_neuro_AF_popmax', 'controls_AF_popmax', 'GTEx_V8_gene','CLNDN', 'ExAC_EAS', 'CLNDISDB', 'ExAC_AMR', 'ExAC_AFR',
        'CLNALLELEID', 'ExAC_NFE', 'ExAC_OTH', 'ExAC_ALL', 'ExAC_SAS', 'ExAC_FIN', 'CLNREVSTAT','LRT_pred','DEOGEN2_pred','MutationAssessor_pred'
        ,'PrimateAI_pred', 'ClinPred_pred', 'MPC_score', 'SIFT4G_pred', 'LIST-S2_pred', 'non_neuro_AF_popmax', 'REVEL_pred',
        'FATHMM_pred', 'PROVEAN_pred','SIFT_score','SIFT_pred','MutationAssessor_score',
        'MutationAssessor_score_rankscore', 'FATHMM_score', 'FATHMM_converted_rankscore', 'PROVEAN_score',
        'PROVEAN_converted_rankscore','DANN_score','integrated_fitCons_score'
    ]
    # 删除这些列
    data = data.drop(columns=columns_to_drop, errors='ignore')

    # 转换 CLINSIG 列中的值
    def convert_clinsig(value):
        if 'Likely_benign' in value or 'Benign' in value or 'Benign/Likely_benign' in value:
            return '0'
        elif 'Likely_pathogenic' in value or 'Pathogenic' in value or 'Pathogenic/Likely_pathogenic' in value:
            return '1'
        return value  # 如果既不是benign也不是pathogenic，则保持原值

    # 应用转换函数到 CLINSIG 列
    if 'CLNSIG' in data.columns:
        data['CLNSIG'] = data['CLNSIG'].apply(convert_clinsig)
    else:
        print("Warning: 'CLNSIG' column not found in the data.")

    # 转换 CLINSIG 列中的值
    def convert(value):
        if value == '.' or isinstance(value, float):
            return None
        elif 'D' in value or 'A' in value:
            return '1'
        else:
            return '0'

    list = ['MetaSVM_pred', 'MetaLR_pred', 'M-CAP_pred','MutationTaster_pred']
    data[list] = data[list].applymap(convert)

    def convert_CADD(value):
        if value == '.':
            return None
        if float(value) > 30:
            return '1'
        else:
            return '0'

    data['CADD_pred'] = data['CADD_phred'].apply(convert_CADD)

    def convert_DANN(value):
        if value == '.':
            return None
        if float(value) > 0.5:
            return '1'
        else:
            return '0'

    data['DANN_pred'] = data['DANN_rankscore'].apply(convert_DANN)

    def convert_VEST4(value):
        if value == '.':
            return None
        if float(value) > 0.5:
            return '1'
        else:
            return '0'

    data['VEST4_pred'] = data['VEST4_score'].apply(convert_VEST4)

    def convert_mut(value):
        if value == '.' or value == '-':
            return None
        if float(value) > 0.75:
            return '1'
        else:
            return '0'

    data['MutPred_pred'] = data['MutPred_score'].apply(convert_mut)

    def convert_mvp(value):
        if value == '.' or value == '-':
            return None
        if float(value) > 0.7:
            return '1'
        else:
            return '0'

    data['MVP_pred'] = data['MVP_score'].apply(convert_mvp)

    def convert_revel(value):
        if value == '.' or value == '-':
            return None
        if float(value) > 0.5:
            return '1'
        else:
            return '0'

    data['REVEL_pred'] = data['REVEL_score'].apply(convert_revel)
    list_score = ['MetaSVM_score', 'MetaLR_score', 'M-CAP_score', 'DANN_rankscore', 'MutPred_score', 'MVP_score',
                  'CADD_phred', 'MutationTaster_score', 'VEST4_score','REVEL_score']

    def convert_score(value):
        if value == '.' or value == '-':
            return None
        else:
            return value

    data[list_score] = data[list_score].applymap(convert_score)

    return data


def reorder_columns(data):

    # 定义列的新顺序，这里只展示部分，你可以根据需要添加更多
    new_order = [
        # 基础信息9
        'Chr','Start','End','Ref','Alt','Func.refGene','Gene.refGene','GeneDetail.refGene',
        'AAChange.refGene',

        # 其他工具预测20
        'MetaSVM_pred', 'MetaLR_pred', 'M-CAP_pred','DANN_pred','MutPred_pred','MVP_pred','REVEL_pred',
        'CADD_pred','MutationTaster_pred','VEST4_pred',
        'MetaSVM_score', 'MetaLR_score', 'M-CAP_score', 'DANN_rankscore','MutPred_score','MVP_score',
        'CADD_phred', 'MutationTaster_score', 'VEST4_score','REVEL_score',

        # 人群等位基因频率13
        'AF', 'AF_raw','AF_male', 'AF_female', 'AF_afr', 'AF_ami', 'AF_amr',
        'AF_asj', 'AF_eas', 'AF_fin', 'AF_nfe', 'AF_oth', 'AF_sas',

        # 功能预测分数6
        'gdi', 'gdi_phred', 'rvis1', 'rvis2', 'lof_score','func',
        # 剪切8
        'DS_AG', 'DS_AL', 'DS_DG','DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',
        # 保守性评分17
        'GERP++_NR','GERP++_RS','phyloP100way_vertebrate','phyloP30way_mammalian','phyloP17way_primate',
        'phastCons100way_vertebrate','phastCons30way_mammalian','phastCons17way_primate',
        'GERP++_RS_rankscore','phyloP100way_vertebrate_rankscore','phyloP20way_mammalian',
        'phyloP20way_mammalian_rankscore','phastCons100way_vertebrate_rankscore',
        'phastCons20way_mammalian','phastCons20way_mammalian_rankscore','SiPhy_29way_logOdds',
        'SiPhy_29way_logOdds_rankscore',

        # 表征17
        'molecular_weight',
        'equipotential_point', 'hydrophilic', 'hydrophobic', 'amphipathic ', 'cyclic',
        'essential', 'aromatic', 'aliphatic', 'nonpolar', 'polar_uncharged', 'acidic',
        'basic', 'sulfur', 'pka_cooh', 'pka_nh3', 'blosum100',
        # 生化9
        'Gm12878', 'H1hesc', 'Hepg2',
        'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf'
    ]

    label_column = ['CLNSIG']
    new_order = new_order + phenotype_columns+ label_column
    if set(new_order) != set(data.columns):
        print("Warning: Column mismatch detected")
        missing = set(data.columns) - set(new_order)
        additional = set(new_order) - set(data.columns)
        if missing:
            print("Missing columns in new_order:", missing)
        if additional:
            print("Additional columns in new_order:", additional)

    data = data[new_order]
    return data


def replace_empty_with_zero(data):

    data.replace({'': np.nan, '.': np.nan}, inplace=True)
    num_cols = data.shape[1] - 30
    print(num_cols)
    cols = ['Chr', 'Start', 'End', 'Ref', 'Alt', 'Func.refGene', 'Gene.refGene', 'GeneDetail.refGene',
           'AAChange.refGene', 'CLNSIG', 'MetaSVM_pred', 'MetaLR_pred', 'M-CAP_pred', 'DANN_pred', 'MutPred_pred',
           'MVP_pred', 'REVEL_pred',
           'CADD_pred', 'MutationTaster_pred', 'VEST4_pred',
           'MetaSVM_score', 'MetaLR_score', 'M-CAP_score', 'DANN_rankscore', 'MutPred_score', 'MVP_score',
           'CADD_phred', 'MutationTaster_score', 'VEST4_score', 'REVEL_score']
    non_chr_start_cols = [col for col in data.columns if col not in cols]
    non_null_count = data[non_chr_start_cols].apply(lambda row: sum(pd.notnull(cell) for cell in row), axis=1)
    non_null_rate_per_row = non_null_count / num_cols
    threshold = 0
    data = data[non_null_rate_per_row >= threshold]

    return data
input_csv = './train/merged_train.csv'
output_csv = './train/train_no_0.csv'
data = pd.read_csv(input_csv,low_memory=False)
phenotype_columns = list(data.columns[196:])
print(len(phenotype_columns))
data = process_csv(data)
categorical_columns = ['func','blosum100','Gm12878', 'H1hesc', 'Hepg2','Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
        'hydrophilic', 'hydrophobic', 'amphipathic ', 'cyclic',
        'essential', 'aromatic', 'aliphatic', 'nonpolar', 'polar_uncharged', 'acidic','basic',
       'sulfur']

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

data = reorder_columns(data)
X = data.iloc[:, 99:-1]
y = data['CLNSIG']
print(X.columns)

pca = SparsePCA(n_components=10, random_state=42)
with open('./train_hpo.json', 'w') as f:
    json.dump(list(X.columns), f)
pca.fit(X)
X_pca = pca.transform(X)
joblib.dump(pca, 'pca.gz')
data = pd.concat([data.iloc[:, :99], pd.DataFrame(X_pca),data.iloc[:,-1]], axis=1)
data = replace_empty_with_zero(data)
data.columns = data.columns.astype(str)
print(data['CLNSIG'].value_counts())

imputer = SimpleImputer(strategy='constant', fill_value=0)
data_to_impute = data.iloc[:,29:-1]
print(data_to_impute.columns)

data_to_impute.columns = data_to_impute.columns.astype(str)

data_imputed = imputer.fit_transform(data_to_impute)
joblib.dump(imputer, 'imputer.gz')
data_imputed = pd.DataFrame(data_imputed, columns=data_to_impute.columns)
data = pd.concat([data.iloc[:,:29],data_imputed, data[['CLNSIG']]], axis=1)
data.to_csv(output_csv, index=False)