from configs import *
from model import *
from data import *
from data_kmr import *
from utils import *
from color_proc import *
from tqdm import tqdm
import segmentation_models as sm
from sklearn.metrics import roc_curve, auc
sm.set_framework("tf.keras")


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cvid",type=int)
args = parser.parse_args()

cvid = args.cvid
# cvid=0
# data_name = "G123-57-e2e-cv%s"%(cvid//3)
data_name = "ALL"

tr_ids,val_ids = tr_val_config(data_name, fold, cvid)
configstring = "%s_%s_%s_%s_%d_lr%s_bs%sxn%s" % (
    framework,
    model_name,
    data_name,
    loss_name,
    edge_size,
    lrstr,
    bs,
    num_gpus,
)
print(configstring)


model_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+{epoch:02d}.h5" % (
    framework,
    model_name,
    data_name,
    loss_name,
    edge_size,
    lrstr,
    continue_step[1] + continue_step[0],
)

continue_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5" % (
    framework,
    model_name,
    data_name,
    loss_name,
    edge_size,
    lrstr,
    continue_step[0],
    continue_step[1],
)

logdir = (
    HOME_PATH
    + "/logs/scalars/"
    + datetime.now().strftime("%Y%m%d-%H%M%S")
    + configstring
)

def log_val_images():
    pass

if not flag_test:
    trainGene_n, valGene_n = load_kmr57_tfdata(
        dataset_path=train_path,
        batch_size=bs,
        cross_fold=["*"],
        # cross_fold = cross_fold[0],
        wsi_ids=tr_ids,
        # wsi_ids=['7015','1052','3768','5256','6747','8107','2502','7930','7885'],
        # wsi_ids=['*'],
        stains=["Tensor/HEpng", "IHC"],  # DAB, Mask, HE< IHC
        aug=False,
        target_size=target_size,
        cache=False,
        shuffle_buffer_size=5000,
        seed=seed,
        num_shards=1,
    )
    if not ("ALL" in data_name):
        valGene_n, _ = load_kmr57_tfdata(
            dataset_path=train_path,
            batch_size=bs_v,
            # cross_fold=["*"],
            cross_fold = cross_fold[1],
            wsi_ids=val_ids,
            stains=["Tensor/HEpng", "IHC"],  # DAB, Mask, HE< IHC
            aug=False,
            target_size=target_size,
            cache=False,
            shuffle_buffer_size=5000,  
            seed=seed,
            num_shards=1,
        )
    trainGene, n_train = trainGene_n
    valGene, n_val = valGene_n
    # for x, y in trainGene:
    #     surf(y[0,:,:,2], div=(10,10))
    #     plt.figure()
    #     plt.subplot(121);plt.imshow(x[0]);plt.axis(False)
    #     plt.subplot(122);plt.imshow(y[0, :, :, 2], cmap="gray");plt.axis(False)
    #     plt.show()
    #     print(y.numpy().max())
    #     print(y.numpy().min())
    print("NUM TRAIN:", n_train)
    print("NUM VAL:", n_val)
    step_num = n_train // bs
    # step_num=10 # for test

    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor="loss",
        verbose=verbose,
        save_best_only=False,
        save_weights_only=True,
        mode="auto",
        save_freq=checkpoint_period * step_num * oversampling,
    )

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = TensorBoard(log_dir=logdir)
    callbacks = [model_checkpoint, tensorboard_callback]

    # model = deeplab(loss=loss_name, lr = lr, classes=3)
    model = smunet(loss=loss_name)
    # model = u_res50enc(loss=loss_name)
    # model = unet(loss=loss_name,
    #  	pl=[16,32,64,128,256],
    #  	# pl=[256,128,64,32,16], # reverted
    #  	)
    # model = kumatt(loss=loss_name)
    training_history = model.fit(
        trainGene,
        validation_data=valGene,
        validation_freq=1,
        validation_steps=n_val // bs_v,
        steps_per_epoch=step_num * oversampling,
        epochs=num_epoches,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )
else:
    model = smunet(loss=loss_name)

    # ----------------------------------------------------------------------
    # 转换tfpb模型为h5
    # ----------------------------------------------------------------------

    # for k in range(10, 11):
    #     start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5" % (
    #         framework,
    #         model_name,
    #         data_name,
    #         loss_name,
    #         edge_size,
    #         lrstr,
    #         continue_step[0] + continue_step[1],
    #         k * checkpoint_period,
    #     )
    #     print(start_path)
    #     model = tf.keras.models.load_model(start_path)
    #     model.save(model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5"% (
    #         framework,
    #         model_name,
    #         data_name,
    #         loss_name,
    #         edge_size,
    #         lrstr,
    #         continue_step[0] + continue_step[1],
    #         k * checkpoint_period,
    #     ))

    avgiou = 0
    fig = plt.figure(figsize=(10,10), dpi=300)
    plt.plot([0, 1], [0, 1], "k--")
    auclist = []
    k_val_global=0
    for id_loocv_t in range(0,1):
        data_name_t = data_name
        if "cv" in data_name:
            cvid = int(id_loocv_t*3)
        else:
            cvid = int(id_loocv_t)
        sing_group = permute_sing(6)
        sing = sing_group[cvid % 6]

        if "sing" in data_name:
            start_path = (
                "/wd_0/ji/_MODELS/sing/"
                + "hvd-tfk-dense121-unet__G123-57-sing%s_bceja_256_lr1.00e-03_ep00+05.h5"
                % id_loocv_t
            )
            tr_ids = list(foldmat[cvid // 6][:3*sing[0]]) + list(foldmat[cvid // 6][3*(sing[0]+1):])
            val_ids = foldmat[cvid // 6][3*sing[0]:3*(sing[0]+1)]
            if cvid%6 == 5:
                tr_ids = list(foldmat[cvid // 6][:3*sing[0]])
                val_ids = foldmat[cvid // 6][3*sing[0]:]
        elif "cv" in data_name:
            start_path = (
                "/wd_0/ji/_MODELS/230405/ccv/intv1/HEOD-IHCRGB/"
                + "tfk-dense121-unet-e2e-odrgb-intv1__G123-57-e2e-cv%s_l1_256_lr1.00e-03_ep00+50.h5"
                % id_loocv_t
            )
            tr_ids= fold["G1"][:cvid] + fold["G2"][:cvid] + fold["G3"][:cvid] + fold["G1"][cvid+3:] + fold["G2"][cvid+3:] + fold["G3"][cvid+3:]
            val_ids = fold["G1"][cvid:cvid+3] + fold["G2"][cvid:cvid+3] + fold["G3"][cvid:cvid+3]
            if cvid//3 == 5:
                tr_ids = fold["G1"][:cvid] + fold["G2"][:cvid] + fold["G3"][:cvid]
                val_ids = fold["G1"][cvid:] + fold["G2"][cvid:] + fold["G3"][cvid:]
        elif "xg" in data_name:
            start_path = (
                model_dir
                + "hvd-tfk-dense121-unet__G123-57-xg%s_bceja_256_lr1.00e-03_ep00+10.h5"
                % id_loocv_t
            )
            G_list = ["G1", "G2", "G3"]
            xg_group = permute_sing(3)
            tr_ids = fold[G_list[xg_group[cvid][1]]][:] + fold[G_list[xg_group[cvid][2]]][:] 
            val_ids= fold[G_list[xg_group[cvid][0]]][:]
        elif "ALL" in data_name:
            # if "e2e" in data_name:
            start_path = (
            "/wd_0/ji/_MODELS/230405/intra/intv1/HERGB-IHCMask/"
            + "tfk-dense121-unet-e2e-rgbmask-intv1__ALL_bceja_256_lr1.00e-03_ep00+50.h5"
            )
    
            tr_ids = ["*"]
            val_ids = ["*"]
        print(">>>>\n", start_path)
        model.load_weights(start_path)
        print(start_path)
        print(tr_ids)
        print(val_ids)


        #--------------------------- test -------------------------------

        testGene, n_test = glob_kmr_test(
            dataset_path=test_path,
            target_size=(2048, 2048),
            batch_size=1,
            cross_fold=None,
            wsi_ids = val_ids,
            aug=False,
            cache=False,
            shuffle_buffer_size=128,
            stains=["HE", "IHC"],
            seed=seed,
        )
        for kk, (tx, ty) in zip(range(n_test), testGene):
            txl = tx
            print(txl)
            tx = cv.imread(tx)
            tx = cv.cvtColor(tx, cv.COLOR_BGR2RGB)/255.0
            # tx = rgbdeconv(tx, H_HE_inv) #--------- for OD input
            tx = np.expand_dims(tx, 0)
            res, hema_texture, mask = interactive_prediction(tx[0, :, :, :3], model, e2e=False)
            # res, hema_texture, mask = e2e_pred(tx[0, :, :, :3], model, sep=True)
            res[res>1] = 1
            res[res<0] = 0
            # res = np.dstack((res[:,:,2], res[:,:,1], res[:,:,0])*255
            # plt.hist(res.ravel());plt.show()
            plt.imsave(
                "/wd_0/ji/DATA/tmp/RGBMask/intra/%s.png" % txl.rsplit("/", 1)[-1],
                res,
            )
"""
        #--------------------------- val -------------------------------
        if not ("ALL" in data_name):
            valGene_n, _ = load_kmr57_tfdata(
                dataset_path=train_path,
                batch_size=bs,
                cross_fold=["*"],
                wsi_ids=val_ids,
                stains=["HE", "IHC"],  # DAB, Mask, HE< IHC
                aug=False,
                target_size=target_size,
                cache=False,
                shuffle_buffer_size=5000,  
                seed=seed,
                num_shards=1,
            )
        else:
            _, valGene_n = load_kmr57_tfdata(
                dataset_path=train_path,
                batch_size=bs,
                cross_fold=["*"],
                wsi_ids=tr_ids,
                stains=["HE", "IHC"],  # DAB, Mask, HE< IHC
                aug=False,
                target_size=target_size,
                cache=False,
                shuffle_buffer_size=5000,
                seed=seed,
                num_shards=1,
            )
        valGene, n_val = valGene_n
        for k_val, (x, y) in zip( tqdm(range(n_val//bs_v)), valGene):
            f = model.predict(x, batch_size=bs_v)
            y = y.numpy().reshape(-1,)[::10]
            f = f.reshape(-1,)[::10]
            if k_val_global == 0:
                Y, F = y, f
            else:
                Y, F = np.concatenate([Y, y]), np.concatenate([F, f])
            k_val_global += 1
    print(classification_report(Y > 0, F>0.5))
    # np.savetxt("%s.csv"%cases, [Y, F], delimiter=",")
    fpr, tpr, _ = roc_curve(Y.ravel(), F.ravel())
    area_under_curve = auc(fpr, tpr)
    auclist.append(area_under_curve)
    plt.plot(fpr, tpr, label="AUC = {:.3f}".format(area_under_curve))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    legs = ['Luck']+['All cases']
    # legs = ['Luck']+[x[6:10] for x in foldmat.ravel()]
    for k in range(1, len(legs)): legs[k] += ", {:.3f}".format(auclist[k-1])
    plt.legend(legs, loc="best")
    plt.tight_layout()
    plt.grid()
    # plt.show()
    plt.savefig("/home/cunyuan/roc.png")
        # plt.imsave(
        #     "/home/cunyuan/resdenseunet/he_%d.png"
        #     % kk,
        #     tx.numpy()[:,:,:,:3].reshape(2048, 2048, 3),
        # )
"""