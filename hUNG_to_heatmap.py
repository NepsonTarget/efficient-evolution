# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:10:24 2023

@author: Nepson
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


pro_len= len('FFSPSPARKRHAPSPEPAVQGTGVAGVPEESGDAAAIPAKKAPAGQEEPGTPPSSPLSAEQLDRIQRNKAAALLRLAARNVPVGFGESWKKHLSGEFGKPYFIKLMGFVAEERKHYTVYPPPHQVFTWTQMCDIKDVKVVILGQDPYHGPNQAHGLCFSVQRPVPPPPSLENIYKELSTDIEDFVHPGHGDLSGWAKQGVLLLNAVLTVRAHQANSHKERGWEQFTDAVVSWLNQNSNGLVFLLWGSYAQKKGSAIDRKRHHVLQTAHPSPLSVYRGFFGCRHFSKTNELLQKSGKKPIDWKEL')

aa_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}
UR50S_array = np.zeros((20,pro_len))
UR90S_1_array = np.zeros((20,pro_len))
UR90S_2_array = np.zeros((20,pro_len))
UR90S_3_array = np.zeros((20,pro_len))
UR90S_4_array = np.zeros((20,pro_len))
UR90S_5_array = np.zeros((20,pro_len))

with open(r"hUNG_alpha.txt", 'r') as f:
    line = f.readline()
    while line:
        fields = line.split('\t')
        mutation = fields[0]
        prob = float(fields[1])
        model = fields[2][1:-2]
        pos_c = int(mutation[1:-1]) - 1
        pos_r = aa_dict[mutation[-1]]
        if model == '\'esm1b_t33_650M_UR50S\'':
            UR50S_array[pos_r, pos_c] = prob
        elif model == '\'esm1v_t33_650M_UR90S_1\'':
            UR90S_1_array[pos_r, pos_c] = prob
        elif model == '\'esm1v_t33_650M_UR90S_2\'':
            UR90S_2_array[pos_r, pos_c] = prob
        elif model == '\'esm1v_t33_650M_UR90S_3\'':
            UR90S_3_array[pos_r, pos_c] = prob
        elif model == '\'esm1v_t33_650M_UR90S_4\'':
            UR90S_4_array[pos_r, pos_c] = prob
        elif model == '\'esm1v_t33_650M_UR90S_5\'':
            UR90S_5_array[pos_r, pos_c] = prob
        line = f.readline()

def build_heatmap(arr,pro_len,fig_name):
    index = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    df = pd.DataFrame(arr,index=index, columns=[i for i in range(1,pro_len+1)])
    df = df.loc[:,(df!=0).any(axis=0)]
    plt.figure(figsize=(20,5),dpi=120)
    sns.heatmap(data=df, cmap=plt.get_cmap('Greens'),annot=True)
    plt.savefig(fig_name)

build_heatmap(UR50S_array, pro_len,'UR50S.png')    
build_heatmap(UR90S_1_array, pro_len,'UR90S_1.png') 
build_heatmap(UR90S_2_array, pro_len,'UR90S_2.png')       
build_heatmap(UR90S_3_array, pro_len,'UR90S_3.png')      
build_heatmap(UR90S_4_array, pro_len,'UR90S_4.png')         
build_heatmap(UR90S_5_array, pro_len,'UR90S_5.png')      
        
     
        
     
        
     
        
     
        
     
        
     