# from kdeeplabv3.model import Deeplabv3
import numpy as np
import os

import skimage.io as io
import skimage.transform as trans
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import losses, callbacks

# import tensorflow_io as tfio
from tensorflow.keras.models import *
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import segmentation_models as sm

model_dict = {}
loss_dict = {
    "bceja": sm.losses.bce_jaccard_loss,
    "focal10ja": sm.losses.BinaryFocalLoss(alpha=0.02, gamma=100)
    + sm.losses.jaccard_loss,
    "ja": sm.losses.jaccard_loss,
    "focal": sm.losses.BinaryFocalLoss(alpha=0.1, gamma=2),
    "bce": sm.losses.binary_crossentropy,
    "focalja": sm.losses.binary_focal_jaccard_loss,
    "focaldice": sm.losses.binary_focal_dice_loss,
    "dice": sm.losses.dice_loss,
    "bcedice": sm.losses.bce_dice_loss,
    "l1": losses.mean_absolute_percentage_error,
    "l2": losses.mean_squared_error,
}

#%%
def smunet(loss="focal", pretrained_weights=None):
    gpu_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICES, cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with gpu_strategy.scope():
        model = sm.Unet(
            backbone_name="densenet121",
            input_shape=(256, 256, 3),
            classes=3,
            activation="sigmoid",
            weights=None,
            encoder_weights="imagenet",
            encoder_freeze=False,
            encoder_features="default",
            decoder_block_type="upsampling",
            decoder_filters=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
        )

    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model
