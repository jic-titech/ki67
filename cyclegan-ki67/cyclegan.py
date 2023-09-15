# %%
import os
from folds import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
data_path = "/gs/hs0/tga-yamaguchi.m/ji/TILES_FULL_256_intv1/"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cvid", type=int)
args = parser.parse_args()

cvid = int(args.cvid * 3)
data_name = "ALL"

tr_ids, val_ids = tr_val_config(data_name, fold, cvid)

print(tr_ids, val_ids)

#%%
import segmentation_models as sm

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras

import horovod.tensorflow.keras as hvd
# %%

from tensorflow.keras.datasets import mnist
from tensorflow_addons.layers import (
    InstanceNormalization,
)
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    Concatenate,
    Lambda,
    Add,
    MaxPool2D,
)
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import datetime
import sys
from data_loader import load_kmr57_tfdata
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)
print(hvd.local_rank())
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

# %%
num_epoches = 51
bs = 4


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if hvd.rank()==0:
    train_log_dir = 'logs/' + current_time + '/train'
    # test_log_dir = 'logs/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

gamma_init = 0.5
gamma = K.variable(gamma_init)

class CycleGAN:
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.dataset_name = "wsis"
        
        if data_name == "ALL":
            tv_split = 0.9
        else:
            tv_split = 1.0
        trainGene_n, _ = load_kmr57_tfdata(
            dataset_path=data_path,
            batch_size=bs,
            cross_fold=["*"],
            wsi_ids=tr_ids,
            stains=["HE", "IHC"],
            aug=False,
            target_size=(self.img_rows, self.img_cols),
            cache=False,
            shuffle_buffer_size=50000,
            seed=1,
            num_shards=1,
            tv_split = tv_split
        )
        if data_name == "ALL":
            valGene_n = _
        else:
            valGene_n, _ = load_kmr57_tfdata(
                dataset_path=data_path,
                batch_size=bs,
                cross_fold=["*"],
                wsi_ids=val_ids,
                stains=["HE", "IHC"],
                aug=False,
                target_size=(self.img_rows, self.img_cols),
                cache=False,
                shuffle_buffer_size=50000,
                seed=1,
                num_shards=1,
                tv_split = tv_split
            )
        self.trainGene, self.n_train = trainGene_n
        self.valGene, self.n_val = valGene_n
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 16
        self.df = 16

        # Loss weights
        self.lambda_cycle = 5  # Cycle-consistency loss
        self.lambda_id = 1   # Identity loss

        optimizer = tf.optimizers.Adam(1E-4, 0.5)
        optimizer = hvd.DistributedOptimizer(optimizer)
        
        # with gpu_strategy.scope():
            # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A._name = "d_A"
        self.d_B._name = "d_B"
        self.d_A.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
        self.d_B.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_AB._name = "g_AB"
        self.g_BA = self.build_generator()
        self.g_BA._name = "g_BA"

        # Input images from both domains
        img_A = Input(shape=self.img_shape, name="img_A")
        img_B = Input(shape=self.img_shape, name="img_B")

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators

        self.combined = Model(
            inputs=[img_A, img_B],
            outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id],
        )
        self.combined.compile(
            loss=[
                "mse",
                "mse",
                "mae",
                "mae",
                "mae",
                "mae",
            ],
            loss_weights=[
                # gamma.numpy(),
                # gamma.numpy(),
                1.0,
                1.0,
                self.lambda_cycle,
                self.lambda_cycle,
                self.lambda_id,
                self.lambda_id,
            ],
            optimizer=optimizer,
        )

    def build_generator(self, ifsm=True):
        if not ifsm:
            """U-Net Generator"""

            def conv2d(layer_input, filters, f_size=4):
                """Layers used during downsampling"""
                d = Conv2D(filters, kernel_size=f_size, strides=2, padding="same")(
                    layer_input
                )
                d = LeakyReLU(alpha=0.2)(d)
                # d = MaxPool2D()(d)
                d = InstanceNormalization()(d)
                return d

            def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
                """Layers used during upsampling"""
                u = UpSampling2D(size=2)(layer_input)
                u = Conv2D(
                    filters,
                    kernel_size=f_size,
                    strides=1,
                    padding="same",
                    activation="relu",
                )(u)
                if dropout_rate:
                    u = Dropout(dropout_rate)(u)
                u = InstanceNormalization()(u)
                u = Concatenate()([u, skip_input])
                return u

            def resblock(layer_input, filters, f_size=3):
                r = Conv2D(filters, kernel_size=f_size, strides=1, padding="same")(
                    layer_input
                )
                r = LeakyReLU(alpha=0.2)(r)
                r = InstanceNormalization()(r)
                r = Conv2D(filters, kernel_size=f_size, strides=1, padding="same")(r)
                r = LeakyReLU(alpha=0.2)(r)
                r = Add()([r, layer_input])
                return InstanceNormalization()(r)

            def resblockn(n, layer_input, filters, f_size=3):
                x = layer_input
                for k in range(n):
                    x = resblock(x, filters, f_size)
                return x
        
            # Image input
            d0 = Input(shape=self.img_shape)

            # Downsampling
            d1 = conv2d(d0, self.gf)
            d2 = conv2d(d1, self.gf*2)
            d3 = conv2d(d2, self.gf*4)
            d4 = conv2d(d3, self.gf*8)
            d5 = conv2d(d4, self.gf*8)
            d6 = conv2d(d5, self.gf*8)
            d7 = conv2d(d6, self.gf*16)

            # Upsampling
            u1 = deconv2d(d7, d6, self.gf*16)
            u2 = deconv2d(u1, d5, self.gf*8)
            u3 = deconv2d(u2, d4, self.gf*8)
            u4 = deconv2d(u3, d3, self.gf*4)
            u5 = deconv2d(u4, d2, self.gf*2)
            u6 = deconv2d(u5, d1, self.gf)

            u7 = UpSampling2D(size=2)(u6)
            output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

            return Model(d0, output_img)
        else:
            model = sm.Unet(
                backbone_name="densenet121",
                input_shape=(256, 256, 3),
                classes=3,
                activation="tanh",
                weights=None,
                encoder_weights="imagenet",
                encoder_freeze=False,
                encoder_features="default",
                decoder_block_type="upsampling",
                decoder_filters=(256, 128, 64, 32, 16),
                decoder_use_batchnorm=True,
            )
            return model

    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, dlnum=None, normalization=True):
            """Discriminator layer"""
            d = Conv2D(
                filters,
                kernel_size=f_size,
                strides=2,
                padding="same",
                name="feature-layer%s" % dlnum,
            )(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, dlnum=0, normalization=False)
        d2 = d_layer(d1, self.df * 2, dlnum=1)
        d3 = d_layer(d2, self.df * 4, dlnum=2)
        d4 = d_layer(d3, self.df * 8, dlnum=3)

        validity = Conv2D(
            1, kernel_size=4, strides=1, padding="same", name="vali-layer"
        )(d4)

        return Model(img, validity)

    def weightedCycleLoss(self, x, y, D):
        """
        weighed cycle loss in "better cycles"
        :param x: input
        :param y: F(G(x))
        :param D: discriminator
        :return: loss
        """
        fd = D.get_layer("feature-layer0")

        def wcycl(x, y):
            wcloss = D(x) * (
                gamma * K.mean(-K.abs(D(y) - D(x)) + (1 - gamma) * K.mean(K.abs(y - x)))
            )
            # wcloss = D(x) * (1-gamma) * K.mean(K.abs(y - x))

            return wcloss

        return wcycl

    def id_loss(self, x, y):
        # ssim_loss = DSSIMObjective()

        def id_loss(x, y):
            return tensorflow.keras.losses.mean_absolute_error(x, y)

        return id_loss

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        stp = 0
        for epoch in range(epochs):
            # print(K.get_value(gamma))
            for batch_i, (imgs_A, imgs_B) in zip(range(self.n_train//bs), self.trainGene):
                stp += 1
                # if batch_i == 1: break
                # ----------------------
                #  Train Discriminators
                # ----------------------
                # plt.imshow(imgs_A[0]);plt.savefig("%dA.png"%batch_i)
                # plt.imshow(imgs_B[0]);plt.savefig("%dB.png"%batch_i)
                # plt.imsave("%dB.png"%batch_i, imgs_B[0])
                # Translate images to opposite domain
                fake_B = self.g_AB(imgs_A)
                fake_A = self.g_BA(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch(
                    [imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B]
                )

                elapsed_time = datetime.datetime.now() - start_time

                if batch_i%10 == 0 and hvd.rank()==0:
                    with train_summary_writer.as_default():
                        tf.summary.scalar("D loss", d_loss[0], stp)
                        tf.summary.scalar("D Acc", 100 * d_loss[1], stp)
                        tf.summary.scalar("G loss", g_loss[0], stp)
                        tf.summary.scalar("adv", np.mean(g_loss[1:3]), stp)
                        tf.summary.scalar("recon", np.mean(g_loss[3:5]), stp)
                        tf.summary.scalar("id", np.mean(g_loss[5:6]), stp)
                # print(self.d_A(imgs_A))
                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

            if (epoch + 1) % 5 == 0  and hvd.rank()==0: 
                self.g_AB.save("saved_model/g_AB_%s_%s_ep%s.hdf5" % (data_name, cvid, epoch))
                self.g_BA.save("saved_model/g_BA_%s_%s_ep%s.hdf5" % (data_name, cvid, epoch))
                # self.d_A.save("saved_model/d_A_ep%s.hdf5" % epoch)
                # self.d_B.save("saved_model/d_B_ep%s.hdf5" % epoch)
                # self.combined.save("saved_model/combined_ep%s.hdf5" % epoch)
                print("Epoch %d, model saved."%(epoch+1))
            K.set_value(gamma, K.get_value(gamma) + (epoch / num_epoches)*(1-gamma_init))

    def sample_images(self, epoch, batch_i):
        os.makedirs("images/%s" % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A, imgs_B = next(self.valGene)
        # plt.imshow(imgs_A[0]);plt.savefig("%dA.png"%batch_i)
        # plt.imshow(imgs_B[0]);plt.savefig("%dB.png"%batch_i)
        # Translate images to the other domain
        fake_B = self.g_AB(imgs_A)
        fake_A = self.g_BA(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # print(imgs_A.shape)
        imgs_A = imgs_A*0.5+0.5
        imgs_B = imgs_B*0.5+0.5
        
        fake_B = fake_B*0.5+0.5
        fake_A = fake_A*0.5+0.5
        
        reconstr_A = reconstr_A*0.5+0.5
        reconstr_B = reconstr_B*0.5+0.5
        gen_imgs = [imgs_A[0], fake_B[0], reconstr_A[0], imgs_B[0], fake_A[0], reconstr_B[0]]

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ["Original", "Translated", "Reconstructed"]
        fig, axs = plt.subplots(r, c, figsize=(15, 10))

        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                plt.tight_layout()
                if i == 0:
                    axs[i, j].set_title(titles[j])
                axs[i, j].axis("off")
                cnt += 1
        fig.tight_layout()
        # plt.show()
        fig.savefig("images/%s/%s_%s_%d_%d.png" % (self.dataset_name, data_name, cvid, epoch, batch_i))
        plt.close()


gan = CycleGAN()
gan.train(epochs=num_epoches, batch_size=bs, sample_interval=50)

# %%

# from tensorflow.keras.utils import plot_model

# plot_model(gan.combined, to_file="cyclegan.svg")
# plot_model(gan.g_AB, to_file="gab.svg")
gan.combined.summary()
gan.d_A.summary()
gan.g_AB.summary()

# %%

gan

# %%
