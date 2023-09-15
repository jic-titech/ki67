#%%
import os, glob, shutil as shu
import horovod.tensorflow.keras as hvd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import skimage.io as io
import skimage.transform as trans

from itertools import combinations as comb

from skimage.transform import resize
from glob import glob
import imageio
import cv2 as cv
import tensorflow as tf
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %%
# * 3. Tf.data as input pipeline
def load_kmr57_tfdata(
    dataset_path,
    batch_size=16,
    wsi_ids=None,
    cross_fold=None,
    stains=["HE", "Mask"],
    aug=False,
    target_size=(256, 256),
    seed=1,
    cache=None,
    shuffle_buffer_size=40000,
    num_shards = 1,
    tv_split = 0.9
) -> tuple:
    def parse_image(file_path):
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor

        img = tf.io.decode_png(img, channels=3)

        img = tf.image.convert_image_dtype(img, tf.float32)

        img = tf.image.resize(img, [target_size[0], target_size[1]])
        # resize the image to the desired size.
        if aug:
            img = augment(img)
        img = img/0.5 - 1.0
        return img

    def parse_mask(file_path):
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor

        img = tf.io.decode_png(img, channels=1)

        img = tf.image.convert_image_dtype(img, tf.float32)

        img = tf.image.resize(img, [target_size[0], target_size[1]])
        # resize the image to the desired size.
        if aug:
            img = augment(img)
        img = img/0.5 - 1.0
        
        return img

    def parse_tensor(file_path):
        tens = np.load(file_path)
        # tens = (tens - tens.mean())/(tens.std()+1E-6)
        tens[tens > 1] = 1
        tens[tens < 0] = 0
        # tens = np.dstack((tens[:,:,0], tens[:,:,2]))
        # tens = tens[:,:,2]
        tens = tf.convert_to_tensor(tens, dtype=tf.float32)
        if aug:
            tens = augment(tens)
        # print(">>>>>>",tens.shape)
        tens = tens/0.5 - 1.0
        
        return tens
    
    def prepare_for_training(
        ds, cache=cache, shuffle_buffer_size=shuffle_buffer_size, batch_size=batch_size
    ):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        # ds = ds.shuffle(
        #     buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=False
        # )
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        # `prefetch` lets the dataset fetch batches in the background
        # while the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    data_generator_tr,data_generator_val = {},{}
    for staintype in stains:
        if staintype != "Mask":

            def augment(image, seed=seed):
                # Add 6 pixels of padding
                image = tf.image.resize_with_crop_or_pad(
                    image, target_size[0] + 32, target_size[0] + 32
                )
                # Random crop back to the original size
                image = tf.image.random_crop(
                    image, size=[target_size[0], target_size[0], 3], seed=seed
                )
                image = tf.image.random_brightness(
                    image, max_delta=0.01, seed=seed
                )  # Random brightness
                image = tf.image.random_flip_left_right(image, seed=seed)
                image = tf.image.random_flip_up_down(image, seed=seed)
                return image

        else:
            # only the channels of input are different
            def augment(image, seed=seed):
                # Add 6 pixels of padding
                image = tf.image.resize_with_crop_or_pad(
                    image, target_size[0] + 32, target_size[0] + 32
                )
                # Random crop back to the original size
                image = tf.image.random_crop(
                    image, size=[target_size[0], target_size[0], 1], seed=seed
                )

                image = tf.image.random_flip_left_right(image, seed=seed)
                image = tf.image.random_flip_up_down(image, seed=seed)
                return image

        dir_pattern = [
            dataset_path + "/" + staintype + "/*" + wsi + "*" for wsi in wsi_ids
        ]
        print(">>>>> LOAD: prepraing case pattern matching...")
        # list_ds = tf.data.Dataset.list_files(dir_pattern, shuffle=True, seed=seed)
        glob_list = []
        for x in dir_pattern:
            glob_list += glob(x)
        list_ds = tf.data.Dataset.from_tensor_slices(glob_list)
        list_ds = list_ds.shuffle(
            1000000, seed=0
        )
        print("Cases matched.")
        print(">>>>> LOAD: Sharding data...")
        # -------------- >>
        list_ds = list_ds.shard(num_shards=hvd.size(), index=hvd.rank())
        # -------------- <<
        print(">>>>> Applying DS options...")
        list_ds = list_ds.with_options(options)
        print(dir_pattern, len(list_ds))
        list_ds_tr = list_ds.take(int(len(list_ds) * tv_split)) #0.9
        list_ds_val = list_ds.skip(int(len(list_ds) * tv_split))
        list_ds_tr = list_ds_tr.with_options(options)
        list_ds_val = list_ds_val.with_options(options)
        
        print(list_ds_val)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        if staintype == "HE" or staintype == "IHC" or  staintype == "Tensor/IHCpng" or staintype == "Tensor/HEpng":
            labeled_ds_tr = list_ds_tr.map(parse_image, num_parallel_calls=AUTOTUNE)
            labeled_ds_val = list_ds_val.map(parse_image, num_parallel_calls=AUTOTUNE)
        elif staintype == "Tensor/IHC" or staintype == "Tensor/HE":
            labeled_ds_tr = list_ds_tr.map(lambda item:tf.numpy_function(parse_tensor, [item], tf.float32), num_parallel_calls=AUTOTUNE)
            labeled_ds_val = list_ds_val.map(lambda item:tf.numpy_function(parse_tensor, [item], tf.float32), num_parallel_calls=AUTOTUNE)
        elif staintype == "Mask":
            labeled_ds_tr = list_ds_tr.map(parse_mask, num_parallel_calls=AUTOTUNE)
            labeled_ds_val = list_ds_val.map(parse_mask, num_parallel_calls=AUTOTUNE)
        labeled_ds_tr = labeled_ds_tr.with_options(options)
        labeled_ds_val = labeled_ds_val.with_options(options)

        data_generator_tr[staintype] = prepare_for_training(
            labeled_ds_tr,
            cache=(cache + "_%s_%d.tfcache" % (staintype, 1e10 * np.random.rand()))
            if isinstance(cache, str)
            else cache,
        )
        data_generator_val[staintype] = prepare_for_training(
            labeled_ds_val,
            cache=(cache + "_%s_%d.tfcache" % (staintype, 1e10 * np.random.rand()))
            if isinstance(cache, str)
            else cache,
        )
    train_generator = zip(data_generator_tr[stains[0]], data_generator_tr[stains[1]])
    validation_generator = zip(data_generator_val[stains[0]], data_generator_val[stains[1]])
    return ((train_generator, len(list_ds_tr)), (validation_generator, len(list_ds_val)))

# %%
# * 3. Tf.data as input pipeline
def load_kmr_test(
    dataset_path,
    batch_size=1,
    wsi_ids=None,
    cross_fold=None,
    stains=["HE", "IHC"],
    aug=False,
    target_size=(2048, 2048),
    seed=1,
    cache=None,
    shuffle_buffer_size=128,
) -> tuple:
    def parse_image(file_path):
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor

        img = tf.io.decode_png(img, channels=3)

        img = tf.image.convert_image_dtype(img, tf.float32)

        img = tf.image.resize(img, [target_size[0], target_size[1]])
        # resize the image to the desired size.
        return img

    # def parse_tiff(file_path):
    #     img = tf.io.read_file(file_path)
    #     # convert the compressed string to a 3D uint8 tensor
    #     img = tfio.experimental.image.decode_tiff(img)

    #     img = tf.image.convert_image_dtype(img, tf.float32)

    #     img = tf.image.resize(img, [target_size[0], target_size[1]])
    #     # resize the image to the desired size.
    #     return img

    def prepare_for_training(
        ds, cache=cache, shuffle_buffer_size=shuffle_buffer_size, batch_size=batch_size
    ):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    data_generator = {}
    for staintype in stains:
        dir_pattern = [
            dataset_path
            + "/"            
            + staintype
            + "/"
            + "*"
        ]

        list_ds = tf.data.Dataset.list_files(dir_pattern, shuffle=False, seed=seed)
        # list_ds = list_ds.shard(num_shards=hvd.size(), index=hvd.rank())
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        labeled_ds = list_ds.map(parse_image, num_parallel_calls=AUTOTUNE)

        data_generator[staintype] = prepare_for_training(
            labeled_ds,
            cache=(cache + "_%s_%d.tfcache" % (staintype, 1e10 * np.random.rand()))
            if isinstance(cache, str)
            else cache,
        )
    train_generator = zip(
        data_generator["HE"],
        data_generator["IHC"],
    )
    n = len(list_ds)
    return (train_generator, n)

#%%
# * 3. Tf.data as input pipeline
def glob_kmr_test(
    dataset_path,
    batch_size=1,
    wsi_ids=None,
    cross_fold=None,
    stains=["HE", "IHC"],
    aug=False,
    target_size=(2048, 2048),
    seed=1,
    cache=None,
    shuffle_buffer_size=128,
) -> tuple:
    data_generator = {}
    for staintype in stains:
        dir_pattern = [
            dataset_path
            + "/"            
            + staintype
            + "/*"
            + wsi
            +"*"
            for wsi in wsi_ids
        ]
        list_ds = []
        for pattern in dir_pattern:
            list_ds += glob(pattern)
        # print(list_ds)
        data_generator[staintype] = sorted(list_ds)
    train_generator = zip(
        data_generator[stains[0]],
        data_generator[stains[1]]
    )
    n = len(list_ds)
    return (train_generator, n)
