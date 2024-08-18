import time
import re
import numpy as np
import pandas as pd
import vcfpy

chr_list = ['chr' + str(chromosome) for chromosome in range(1, 23)] + ['chrX', 'chrY']
func_dict = {'nonsynonymous SNV': 'nonsynonymous SNV', 'synonymous SNV': 'synonymous SNV',
             'stopgain': 'stopgain', 'unknown': 'unknown', 'startloss': 'startloss',
             'stoploss': 'stoploss', 'frameshift substitution': 'frameshift',
             'frameshift insertion': 'frameshift', 'frameshift deletion': 'frameshift',
             'nonframeshift insertion': 'nonframeshift', 'nonframeshift deletion': 'nonframeshift',
             'nonframeshift substitution': 'nonframeshift',
             '.':'unknown'}
type_dict = {'0': 0,
             'Active_Promoter': 1,
             'Heterochrom/lo': 2,
             'Insulator': 3,
             'Poised_Promoter': 4,
             'Repetitive/CNV': 5,
             'Repressed': 6,
             'Strong_Enhancer': 7,
             'Txn_Elongation': 8,
             'Txn_Transition': 9,
             'Weak_Enhancer': 10,
             'Weak_Promoter': 11,
             'Weak_Txn': 12,
             0: 0}
spliceai_column_list = ['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']
def spliceAI(vcf_path,df):

    vcf_data = vcfpy.Reader.from_path(vcf_path)
    data_lists = []
    index = 0
    gene_list = df['Gene.refGene'].to_list()
    for record in vcf_data:
        temp_list = []
        if record.INFO.get('SpliceAI') is not None:
            spliceai_list = record.INFO.get('SpliceAI')[0].split(',')
            for spliceai in spliceai_list:
                temp_list = spliceai.split('|')
                if temp_list[1] == gene_list[index]:
                    temp_list = [float(i) if i not in ['nan', '.'] else 0.0 for i in temp_list[2:]]
                else:
                    temp_list = [0.0] * 8
        else:
            temp_list = [0.0] * 8
        data_lists.append(temp_list)
        index += 1
    if df.shape[0] == len(data_lists):
        data_spliceai = pd.DataFrame(np.matrix(data_lists))
        data_spliceai.columns = spliceai_column_list
        df = df.join(data_spliceai, how='left')
        print("============spliceAI融合完成==================")
        return df
    else:
        assert 'Cannot add features to data'
    # 写入结果到新的CSV文件


def annotate_functional_effect(data):
    print('---' + time.asctime(time.localtime(time.time())) + '--- ' + 'annotating functional effects\n')

    data['ExonicFunc.refGene'].fillna('unknown', inplace=True)
    func_list = data['ExonicFunc.refGene'].tolist()
    data.drop('ExonicFunc.refGene', axis=1, inplace=True)
    data.insert(data.shape[-1], 'func', func_list)

    gene_list = data['Gene.refGene'].unique().tolist()
    sum_gdi = {}
    sum_gdi_phred = {}
    all_disease = {}
    all_Mendelian = {}
    mendelian_AD = {}
    mendelian_AR = {}
    all_PID = {}
    pid_AD = {}
    pid_AR = {}
    count = 0
    with open(gdi_file, 'r') as read_GDI:
        while True:
            line = read_GDI.readline()
            if line:
                list_text = line.split()
                if str(list_text[0]) in gene_list:
                    sum_gdi[list_text[0]] = float(list_text[1])
                    sum_gdi_phred[list_text[0]] = float(list_text[2])
                    all_disease[list_text[0]] = list_text[3]
                    all_Mendelian[list_text[0]] = list_text[4]
                    mendelian_AD[list_text[0]] = list_text[5]
                    mendelian_AR[list_text[0]] = list_text[6]
                    all_PID[list_text[0]] = list_text[7]
                    pid_AD[list_text[0]] = list_text[8]
                    pid_AR[list_text[0]] = list_text[9]
                    count += 1
                if count == len(gene_list):
                    break
            else:
                break

    for i in gene_list:
        if i not in sum_gdi.keys():
            sum_gdi[i] = np.nan
            sum_gdi_phred[i] = np.nan
            all_disease[i] = np.nan
            all_Mendelian[i] = np.nan
            mendelian_AD[i] = np.nan
            mendelian_AR[i] = np.nan
            all_PID[i] = np.nan
            pid_AD[i] = np.nan
            pid_AR[i] = np.nan

    count = 0
    rvis1 = {}
    rvis2 = {}
    with open(rvis_file, 'r') as read_RVIS:
        while True:
            line = read_RVIS.readline()
            if line:
                list_text = line.split()
                if list_text[0] in gene_list or list_text[1] in gene_list or \
                        list_text[2] in gene_list or list_text[3] in gene_list or \
                        list_text[4] in gene_list:
                    rvis1[list_text[0]] = float(list_text[5])
                    rvis2[list_text[0]] = float(list_text[6])
                    count += 1
                if count == len(gene_list):
                    break
            else:
                break

    for i in gene_list:
        if i not in rvis1.keys():
            rvis1[i] = np.nan
            rvis2[i] = np.nan

    count = 0
    lof_score = {}
    with open(lof_file, 'r') as read_lof:
        while True:
            line = read_lof.readline()
            if line:
                list_text = line.split()
                if str(list_text[0]) in gene_list:
                    lof_score[str(list_text[0])] = float(list_text[1])
                    count += 1
                if count == len(gene_list):
                    break
            else:
                break

    for i in gene_list:
        if i not in lof_score.keys():
            lof_score[i] = np.nan

    sum_gdi_list = []
    sum_gdi_phred_list = []
    rvis1_list = []
    rvis2_list = []
    lof_score_list = []
    for record in data['Gene.refGene']:
        sum_gdi_list.append(sum_gdi[record])
        sum_gdi_phred_list.append(sum_gdi_phred[record])
        rvis1_list.append(rvis1[record])
        rvis2_list.append(rvis2[record])
        lof_score_list.append(lof_score[record])

    lists = [sum_gdi_list, sum_gdi_phred_list, rvis1_list, rvis2_list, lof_score_list]
    column_name_list = ['gdi', 'gdi_phred', 'rvis1', 'rvis2', 'lof_score']
    column_name_dict = {}
    for col_index in range(len(column_name_list)):
        column_name_dict[col_index] = column_name_list[col_index]
    data_gene = pd.DataFrame(lists).T
    data_gene.rename(columns=column_name_dict, inplace=True)
    data = data.join(data_gene, how='left')
    print("============Function融合完成==================")
    return data

def annotate_biochemical(data):
    print('---' + time.asctime(time.localtime(time.time())) + '--- annotating biochemical properties\n')
    data_aa = pd.read_csv(aachange_file,low_memory=False)
    aa3_dict = {}
    aa1_dict = {}
    aa_name_dict = {}
    for record in data_aa.iterrows():
        aa3_dict[record[1]['3_letter_name']] = record[1].to_list()[3:20]
        aa1_dict[record[1]['1_letter_name']] = record[1].to_list()[3:20]
        aa_name_dict[record[1]['3_letter_name']] = record[1]['1_letter_name']

    data_blosum = pd.read_csv(blosum_file)
    data_blosum = data_blosum.set_index(data_blosum.aa)
    data_blosum.drop('aa', axis=1, inplace=True)

    data = data[data['AAChange.refGene'] != 'UNKNOWN']
    data.index = list(range(data.shape[0]))

    aa_lists = []
    used_nm_list = []
    used_aachange_list = []
    pattern = '[A-Z]\d+[A-Z]'

    for record in data['AAChange.refGene']:
        if str(record) == 'nan':
            aa_lists.append([np.nan] * 16)
            used_nm_list.append(np.nan)
            used_aachange_list.append(np.nan)
            continue
        count = len(record.split(','))
        for i in record.split(','):
            if str(i) == 'nan':
                aa_lists.append([np.nan] * 16)
                used_nm_list.append(np.nan)
                used_aachange_list.append(np.nan)
                break
            else:
                if 'p.' in i and re.match(pattern, i.split('p.')[1]):
                    aa = re.split('\d+', re.match(pattern, i.split('p.')[1]).group())
                    aa_str = aa[0] + aa[1]
                    # see if the aachange contains any ambiguious meaning
                    if 'B' not in aa_str and 'Z' not in aa_str and 'J' not in aa_str and 'X' not in aa_str:
                        aa_list = (np.array(aa1_dict[aa[1]]) - np.array(aa1_dict[aa[0]])).tolist()[:-1]
                        aa_lists.append(aa_list)
                        used_nm_list.append(i.split(':')[1])
                        used_aachange_list.append(i.split('p.')[1])
                        break
                count -= 1
                if count <= 0:
                    aa_lists.append([np.nan] * 16)
                    used_nm_list.append(np.nan)
                    used_aachange_list.append(np.nan)
                    break

    data_aachange = pd.DataFrame(np.matrix(aa_lists))
    data_aachange.columns = data_aa.columns[3:-4]
    data = data.join(data_aachange, how='left')
    data.insert(data.shape[-1], 'used_aachange', used_aachange_list)
    data.insert(data.shape[-1], 'used_nm', used_nm_list)

    blosum_list = []

    for record in data['AAChange.refGene']:
        if str(record) == 'nan':
            blosum_list.append(np.nan)
            continue
        count = len(record.split(','))
        for i in record.split(','):
            if str(i) == 'nan':
                blosum_list.append(np.nan)
                break
            else:
                if 'p.' in i and re.match(pattern, i.split('p.')[1]):
                    aa = re.split('\d+', re.match(pattern, i.split('p.')[1]).group())
                    aa_str = aa[0] + aa[1]
                    # see if the aachange contains any ambiguous meaning
                    if 'J' not in aa_str:
                        blosum_list.append(data_blosum.loc[aa[0], aa[1]])
                        break
                    else:
                        blosum_list.append(np.nan)
                        break
                count -= 1
                if count <= 0:
                    blosum_list.append(np.nan)
                    break
    data.insert(data.shape[-1], 'blosum100', blosum_list)
    print("============bio融合完成==================")
    return data


def HPO_files(txt_path, csv_data):
    hpo_data = pd.read_csv(txt_path, sep='\t', dtype=str)
    unique_hpo_ids = hpo_data['hpo_id'].unique()
    hpo_columns = pd.DataFrame(0, index=csv_data.index, columns=unique_hpo_ids)

    gene_to_hpo = hpo_data.groupby('gene_symbol')['hpo_id'].apply(set).to_dict()

    csv_data = pd.concat([csv_data, hpo_columns], axis=1)

    for gene_symbol, hpos in gene_to_hpo.items():
        indices = csv_data[csv_data['Gene.refGene'] == gene_symbol].index
        for hpo_id in hpos:
            if hpo_id in unique_hpo_ids:
                csv_data.loc[indices, hpo_id] = 1
    print('================hpo合并完成==============')
    return csv_data

def annotate_chromHMM(data):  # after annotating spliceai
    print('---' + time.asctime(time.localtime(time.time())) + '--- annotating chromHMM.\n')

    cell_types = ['Gm12878', 'H1hesc', 'Hepg2', 'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf']

    data.Chr = ['chr' + str(i) for i in data.Chr.tolist() if 'chr' not in str(i)]
    data = data[data.Chr.isin(chr_list)]
    data = data.sort_values(by=['Chr', 'Start', 'End', 'Ref', 'Alt'], ascending=True)

    chr_dict = {}

    content_dict = {}
    for c in chr_list:
        chr_dict[c] = []
        content_dict[c] = []
    for c in chr_list:
        count = 0
        with open(epi_file, 'r') as f_read:
            temp = f_read.readline()
            while True:
                line = f_read.readline()
                if line:
                    line = line.strip().split('\t')
                    if line[0] == str(c) and count == 0:
                        chr_dict[c].extend([int(line[1]), int(line[2])])
                        content_dict[c].append(line[3:])
                    elif line[0] == str(c) and count != 0:
                        chr_dict[c].append(int(line[2]))
                        content_dict[c].append(line[3:])
                    count += 1
                else:
                    break

    epi_list = []
    for c in chr_list:
        for record in data[data.Chr == c].iterrows():
            binary_start_index = binarySearch(chr_dict[c], int(record[1]['Start']))
            end_index = binarySearch(chr_dict[c], int(record[1]['End']))
            if binary_start_index == -1 and end_index == -1:
                epi_list.append([0] * 9)
            elif binary_start_index != -1 and end_index == -1:
                epi_list.append(content_dict[c][binary_start_index])
            elif binary_start_index == -1 and end_index != -1:
                epi_list.append(content_dict[c][end_index])
            elif binary_start_index == end_index:
                epi_list.append(content_dict[c][end_index])
            elif binary_start_index != end_index:
                temp = []
                for index in range(9):
                    if content_dict[c][binary_start_index][index] == content_dict[c][end_index][index]:
                        temp.append(content_dict[c][binary_start_index][index])
                    else:
                        temp.append(0)
                epi_list.append(temp)

    s = set()
    for epi in epi_list:
        s.add(len(epi))

    epi_dict = {}
    for epi in cell_types:
        epi_dict[epi] = []
    for index in range(len(epi_list)):
        count = 0
        for epi in cell_types:
            epi_dict[epi].append(epi_list[index][count])
            count += 1

    for col in cell_types:
        data.insert(data.shape[-1], col, epi_dict[col])

    for col in cell_types:
        data[col] = [type_dict[cell_type] for cell_type in data[col].to_list()]

    data = data.sort_index(ascending = True)
    data.index = list(range(data.shape[0]))
    print("============表征融合完成==================")
    return data


def Clinvar(vcf_path, df):
    vcf_data = vcfpy.Reader.from_path(vcf_path)

    vcf_dict = {}
    for record in vcf_data:
        clnsig_values = record.INFO['CLNSIG']
        clnsig_values = clnsig_values[0]
        print(clnsig_values)
        vcf_dict[(record.CHROM, record.POS)] = clnsig_values
    df['CLNSIG'] = df.apply(lambda row: vcf_dict.get((str(row['Chr']), row['Start']), 0), axis=1)
    print("============标签融合完成==================")
    return df


def Clinvar1(df2, df1):
    df2['class'] = df2['group'].astype(str)
    df2['CLASS'] = df2['class']
    print("映射后的CLASS列：")
    print(df2['CLASS'])
    print(df1['Chr'])
    str(df1['Chr'])
    str(df2['Chr'])
    df2['Chr'] = df2['Chr'].apply(lambda x: 'chr' + x if not x.startswith('chr') else x)
    merged_df = pd.merge(df1, df2[['Chr', 'Start', 'End', 'CLASS']], on=['Chr', 'Start', 'End'], how='left')
    merged_df['CLNSIG'] = merged_df['CLASS']
    merged_df.drop(columns=['CLASS'], inplace=True)
    print("============标签融合完成==================")

    return merged_df

def binarySearch(list1, target):
    left = 0
    right = len(list1) - 1

    while left <= right:
        mid = (right + left) // 2
        if mid + 1 < len(list1) and list1[mid] <= target < list1[mid + 1]:
            return mid
        elif target > list1[mid]:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# 设置文件路径
csv_path = './annovar_data/filtered_train.csv'
vcf_path = './spliceAI_train.vcf'
output_path = './train/merged_train.csv'
gdi_file = './annovar_data/GDI_full_10282015.txt'
rvis_file = './annovar_data/RVIS_ExAC_4KW.txt'
lof_file = './annovar_data/LoFtool_scores.txt'
aachange_file = './annovar_data/amino_acid.csv'
blosum_file = './annovar_data/blosum.csv'
hpo_file = './annovar_data/genes_to_phenotype.txt'
epi_file = './annovar_data/master38.chromhmm.bedg'
def merge(csv_path):
    df = pd.read_csv(csv_path,low_memory=False)
    df = spliceAI(vcf_path,df)
    df = annotate_functional_effect(df)
    df = annotate_biochemical(df)
    df = annotate_chromHMM(df)
    df = HPO_files(hpo_file,df)
    df.to_csv(output_path, index=False)
merge(csv_path)
