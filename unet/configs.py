#%%
from utils import *
import tensorflow as tf
from datetime import datetime
import os
import numpy as np

# HOME_PATH = "/raid/ji/"
# train_path = HOME_PATH + "/DATA/TILES_(256, 256)"
# val_path = HOME_PATH + "/DATA/TILES_(256, 256)"
# test_path = "/raid/ji/DATA" + "/KimuraLIpng/"

HOME_PATH = "/wd_0/ji"
train_path = HOME_PATH + "/TILES_FULL_256_intv1"
val_path = HOME_PATH + "/TILES_FULL_256_intv1"
test_path = "/wd_0/ji/TILES_FULL_2048/test/"

model_dir = HOME_PATH + "/models57/"

seed = 1

# ------------------ 指定训练·测试图像尺寸 ---------------------
edge_size = 256
target_size = (edge_size, edge_size)
test_size = (2048, 2048)

# ------------------ 指定GPU资源 ---------------------
# devices = "0,1,2,3"
devices = ""
os.environ["CUDA_VISIBLE_DEVICES"] = devices
DEVICES=[
"/gpu:%s"%id for id in devices[::2]# ::2 skips puncs in device string
]

num_gpus=len(DEVICES)

lr = 1E-3
lr = lr*num_gpus # 线性scale学习率
# model_name = "dense121-unet-linear-norminput"
# model_name = "dense121-unet-e2e-rgbod-intv1"
# model_name = "dense121-unet-e2e-rgbrgb-intv1"
# model_name = "dense121-unet-e2e-odod-intv1"
model_name = "dense121-unet-e2e-odrgb-intv1"
# model_name = "dense121-unet-e2e-rgbmask-intv1"

# !!! ------------------ 强制设置学习率 ---------------------
lr = 1E-3

lrstr = "{:.2e}".format(lr)

bs_single = 64
if num_gpus == 0: num_gpus = 1
bs = bs_single*num_gpus
bs_v = bs_single*num_gpus

verbose = 1

checkpoint_period = 5

flag_test = 1
flag_multi_gpu = 0

flag_continue = 0
continue_step = (0, 0)  # start epoch, total epochs trained
initial_epoch = continue_step[0] + continue_step[1]

num_epoches = 50


fold = {'G1': ['7015','1052','3768',
'7553','5425','3951',
'2189','3135','3315',
'4863','4565','2670',
'3006','3574','3597',
'3944','1508','0669','1115'],
'G2': ['5256','6747','8107',
'1295','2072','2204',
'3433','7144','1590',
'2400','6897','1963',
'2118','4013','4498',
'0003','2943','3525','2839'],
'G3': ['2502','7930','7885',
'0790','1904','3235',
'2730','7883','3316',
'4640','0003','1883',
'2913','1559','2280',
'6018','2124','8132','2850']}

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
            fold[kf][kk] = kf+"_"+fold[kf][kk]
    foldmat = np.vstack([fold[key] for key in fold.keys()])

    if "sing" in data_name:
        tr_ids = list(foldmat[cvid // 6][:3*sing[0]]) + list(foldmat[cvid // 6][3*(sing[0]+1):])
        val_ids = foldmat[cvid // 6][3*sing[0]:3*(sing[0]+1)]
        if cvid%6 == 5:
            tr_ids = list(foldmat[cvid // 6][:3*sing[0]])
            val_ids = foldmat[cvid // 6][3*sing[0]:]
    elif "cv" in data_name:
        tr_ids= fold["G1"][:cvid] + fold["G2"][:cvid] + fold["G3"][:cvid] + fold["G1"][cvid+3:] + fold["G2"][cvid+3:] + fold["G3"][cvid+3:]
        val_ids = fold["G1"][cvid:cvid+3] + fold["G2"][cvid:cvid+3] + fold["G3"][cvid:cvid+3]
        if cvid//3 == 5:
            tr_ids = fold["G1"][:cvid] + fold["G2"][:cvid] + fold["G3"][:cvid]
            val_ids = fold["G1"][cvid:] + fold["G2"][cvid:] + fold["G3"][cvid:]
    elif "xg" in data_name:
        G_list = ["G1", "G2", "G3"]
        xg_group = permute_sing(3)
        tr_ids = fold[G_list[xg_group[cvid][1]]][:] + fold[G_list[xg_group[cvid][2]]][:] 
        val_ids= fold[G_list[xg_group[cvid][0]]][:]
    elif "ALL" in data_name:
        tr_ids = ["*"]
        val_ids = ["*"]

    tr_ids = list(tr_ids)
    val_ids = list(val_ids)
    print(tr_ids)
    print(val_ids)
    return tr_ids, val_ids

framework = "tfk"


loss_name = "l1"  # focalja, bce, bceja, ja, dice...

# data_name = "G123-57-cv%s"%(cvid//3)
oversampling = 1
# FIXED_STEPS = 1600

test_list=[[6, 11], [12, 17], [0,5], [24, 29], [30, 34], [47, 52], [35, 40], [18, 23], [41, 46]]

cross_fold = [["001", "002", "003", "004", "006", "007", "008", "009"], ["005", "010"]]
