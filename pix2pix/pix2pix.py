from __future__ import print_function, division
import scipy
import os
from folds import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
data_path = "/gs/hs0/tga-yamaguchi.m/ji/TILES_FULL_256_intv1/"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cvid", type=int)
args = parser.parse_args()

cvid = int(args.cvid * 3)
data_name = "cv"

tr_ids, val_ids = tr_val_config(data_name, fold, cvid)

print(tr_ids, val_ids)

import segmentation_models as sm
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras
import horovod.tensorflow.keras as hvd

# from lkeras.datasets import mnist
from tensorflow_addons.layers import (
    InstanceNormalization,
)
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
from tensorflow.keras.layers import (
    BatchNormalization,
    Activation,
    ZeroPadding2D,
    LeakyReLU,
)
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from data_loader import load_kmr57_tfdata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

hvd.init()
if hvd.rank() == 0:
    print("Horovod world size: ", hvd.size())
# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)
print("Horovod local device rank: ", hvd.local_rank())
print("Horovod global device rank: ", hvd.rank(), "\n")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

# %%
num_epochs = 51
bs = 16


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if hvd.rank() == 0:
    train_log_dir = "logs/" + current_time + data_name + "cv%s"%cvid + "/train"
    # test_log_dir = 'logs/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)


class Pix2Pix:
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
        self.gf = 64
        self.df = 64

        optimizer = tf.optimizers.Adam(2e-4, 0.5)
        optimizer = hvd.DistributedOptimizer(optimizer)

        # with gpu_strategy.scope():
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss="mse", optimizer=optimizer, metrics=["accuracy"]
        )

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(
            loss=["mse", "mae"], loss_weights=[1, 100], optimizer=optimizer
        )

    def build_generator(self, ifsm=True):
        if not ifsm:

            def conv2d(layer_input, filters, f_size=4, bn=True):
                """Layers used during downsampling"""
                d = Conv2D(filters, kernel_size=f_size, strides=2, padding="same")(
                    layer_input
                )
                d = LeakyReLU(alpha=0.2)(d)
                if bn:
                    d = BatchNormalization(momentum=0.8)(d)
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
                u = BatchNormalization(momentum=0.8)(u)
                u = Concatenate()([u, skip_input])
                return u

            # Image input
            d0 = Input(shape=self.img_shape)

            # Downsampling
            d1 = conv2d(d0, self.gf, bn=False)
            d2 = conv2d(d1, self.gf * 2)
            d3 = conv2d(d2, self.gf * 4)
            d4 = conv2d(d3, self.gf * 8)
            d5 = conv2d(d4, self.gf * 8)
            d6 = conv2d(d5, self.gf * 8)
            d7 = conv2d(d6, self.gf * 8)

            # Upsampling
            u1 = deconv2d(d7, d6, self.gf * 8)
            u2 = deconv2d(u1, d5, self.gf * 8)
            u3 = deconv2d(u2, d4, self.gf * 8)
            u4 = deconv2d(u3, d3, self.gf * 4)
            u5 = deconv2d(u4, d2, self.gf * 2)
            u6 = deconv2d(u5, d1, self.gf)

            u7 = UpSampling2D(size=2)(u6)
            output_img = Conv2D(
                self.channels,
                kernel_size=4,
                strides=1,
                padding="same",
                activation="tanh",
            )(u7)

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
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding="same")(
                layer_input
            )
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding="same")(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        stp = 0

        for epoch in range(epochs):
            for batch_i, (imgs_B, imgs_A) in zip(
                range(self.n_train // bs), self.trainGene
            ):
                stp += 1

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                if hvd.rank() == 0:
                    if batch_i % 20 == 0:
                        with train_summary_writer.as_default():
                            tf.summary.scalar("D loss", d_loss[0], stp)
                            tf.summary.scalar("D Acc", 100 * d_loss[1], stp)
                            tf.summary.scalar("G loss", g_loss[0], stp)
                    if batch_i % sample_interval == 0:
                        self.sample_images(epoch, batch_i)
            if (epoch + 1) % 5 == 0 and hvd.rank() == 0:
                self.generator.save("saved_model/pix2pix_%s_%s_ep%s.hdf5" % (data_name, cvid, epoch))
                print("Epoch %d, model saved." % (epoch + 1))

    def sample_images(self, epoch, batch_i):
        os.makedirs("images/%s" % self.dataset_name, exist_ok=True)
        r, c = 1, 3

        imgs_B, imgs_A = next(self.valGene)
        fake_A = self.generator(imgs_B)
        fake_A = fake_A * 0.5 + 0.5
        imgs_A = imgs_A * 0.5 + 0.5
        imgs_B = imgs_B * 0.5 + 0.5

        gen_imgs = [imgs_B[0], fake_A[0], imgs_A[0]]

        titles = ["Input", "Generated", "GT"]
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for j in range(c):
            axs[j].imshow(gen_imgs[cnt])
            axs[j].set_title(titles[j])
            axs[j].axis("off")
            cnt += 1
        fig.savefig("images/%s/%s_%s_%d_%d.png" % (self.dataset_name, data_name, cvid, epoch, batch_i))
        plt.close()


if __name__ == "__main__":
    gan = Pix2Pix()
    gan.train(epochs=num_epochs, batch_size=bs, sample_interval=100)
