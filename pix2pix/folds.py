#%%
import os
import numpy as np

fold = {
    "G1": [
        "7015",
        "1052",
        "3768",
        "7553",
        "5425",
        "3951",
        "2189",
        "3135",
        "3315",
        "4863",
        "4565",
        "2670",
        "3006",
        "3574",
        "3597",
        "3944",
        "1508",
        "0669",
        "1115",
    ],
    "G2": [
        "5256",
        "6747",
        "8107",
        "1295",
        "2072",
        "2204",
        "3433",
        "7144",
        "1590",
        "2400",
        "6897",
        "1963",
        "2118",
        "4013",
        "4498",
        "0003",
        "2943",
        "3525",
        "2839",
    ],
    "G3": [
        "2502",
        "7930",
        "7885",
        "0790",
        "1904",
        "3235",
        "2730",
        "7883",
        "3316",
        "4640",
        "0003",
        "1883",
        "2913",
        "1559",
        "2280",
        "6018",
        "2124",
        "8132",
        "2850",
    ],
}


def permute_sing(idx):
    l = np.arange(idx)
    sing_group = [list(l)]
    for k in range(idx):
        l1 = np.zeros(len(l))
        l1[:-1] = l[1:]
        l1[-1] = l[0]
        l = l1.astype(int)
        sing_group.append(list(l))
    return sing_group


def tr_val_config(data_name, fold, cvid):
    for kf in fold.keys():
        for kk in range(len((fold[kf]))):
            fold[kf][kk] = kf + "_" + fold[kf][kk]
    foldmat = np.vstack([fold[key] for key in fold.keys()])

    if "sing" in data_name:
        tr_ids = list(foldmat[cvid // 6][: 3 * sing[0]]) + list(
            foldmat[cvid // 6][3 * (sing[0] + 1) :]
        )
        val_ids = foldmat[cvid // 6][3 * sing[0] : 3 * (sing[0] + 1)]
        if cvid % 6 == 5:
            tr_ids = list(foldmat[cvid // 6][: 3 * sing[0]])
            val_ids = foldmat[cvid // 6][3 * sing[0] :]
    elif "cv" in data_name:
        tr_ids = (
            fold["G1"][:cvid]
            + fold["G2"][:cvid]
            + fold["G3"][:cvid]
            + fold["G1"][cvid + 3 :]
            + fold["G2"][cvid + 3 :]
            + fold["G3"][cvid + 3 :]
        )
        val_ids = (
            fold["G1"][cvid : cvid + 3]
            + fold["G2"][cvid : cvid + 3]
            + fold["G3"][cvid : cvid + 3]
        )
        if cvid // 3 == 5:
            tr_ids = fold["G1"][:cvid] + fold["G2"][:cvid] + fold["G3"][:cvid]
            val_ids = fold["G1"][cvid:] + fold["G2"][cvid:] + fold["G3"][cvid:]
    elif "xg" in data_name:
        G_list = ["G1", "G2", "G3"]
        xg_group = permute_sing(3)
        tr_ids = fold[G_list[xg_group[cvid][1]]][:] + fold[G_list[xg_group[cvid][2]]][:]
        val_ids = fold[G_list[xg_group[cvid][0]]][:]
    elif "ALL" in data_name:
        tr_ids = ["*"]
        val_ids = ["*"]

    tr_ids = list(tr_ids)
    val_ids = list(val_ids)
    return tr_ids, val_ids
