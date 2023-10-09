import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import os
import torch

columns = ['rr','rs', 'ra', 'sr','ss','sa', 'ar', 'as','aa']

def attn_stats(attn):

    # attn: 1 x 60 x 60 (60 b.c. 3 modality x T = 20)
    attn = attn[0]
    dim,_ = attn.shape # dim ,dim
    T = dim // 3
    rtg_idx = np.array( [ i*3 for i in range(T)] )
    state_idx = rtg_idx  + 1
    action_idx = rtg_idx  + 2

    idx_list = [ rtg_idx, state_idx, action_idx]
    new_stats = []
    for i, idx_i in enumerate( idx_list ):
        for j, idx_j in enumerate( idx_list ):
            d = attn[:,:][idx_i,:][:, idx_j]
            if i < j:
                # can't attend modality behind me within the same timestep
                ind = np.diag_indices(d.shape[0])
                d[ind[0], ind[1]] = torch.zeros(d.shape[0])

            new_stats.append(torch.sum(d).item())

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    FILE_NAME = f"./Attention_stats/Attention_stats_{current_time}.csv"
    while os.path.exists(FILE_NAME):
        current_time += "_"
        FILE_NAME = f"./Attention_stats/Attention_stats_{current_time}.csv"

    pd.DataFrame([new_stats], columns = columns).to_csv(FILE_NAME, index = False)
    # for i in range(len(names)):
    #     fig, axes = plt.subplots()
    #     plot_data = data_all[i]
    #     plot_name = names[i]
    #
    #     mask = np.zeros_like(plot_data, dtype = np.bool)
    #     mask[np.triu_indices_from(mask)] = True
    #
    #     sns.heatmap(plot_data, mask = mask, linewidth=0.5, ax = axes)
    #     axes.set_title(plot_name)
    #     y_label, x_label = axes_name[i]
    #     axes.set_xlabel(x_label)
    #     axes.set_ylabel(y_label)
    #     plt.tight_layout()
    #     plt.savefig(f'./{folder}/{plot_name}.png')
