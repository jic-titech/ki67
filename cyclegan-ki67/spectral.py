import numpy as np
from numpy import *
import cv2 as cv
import os, glob
import matplotlib.pyplot as plt
from skimage.color import rgb2hed, hed2rgb, separate_stains, combine_stains
from skimage.exposure import rescale_intensity
from matplotlib.colors import LinearSegmentedColormap
from skimage import data
from mpl_toolkits.mplot3d.axes3d import Axes3D  # --- For 3D plot
from skimage.exposure import rescale_intensity
from cv2 import ximgproc
from pathlib import Path

dpl = {"chips": "*A/",
       "ihcs": "*B/"}

# %%
H_DAB = array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78]
])

H_he = H_DAB.copy()
H_he[2, :] = np.cross(H_DAB[0, :], H_DAB[1, :])

H_ki67 = H_DAB.copy()
H_ki67[1, :] = np.cross(H_DAB[0, :], H_DAB[2, :])

cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['white', 'darkviolet'])
cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white', 'saddlebrown'])
print("Trans. H.E", H_he)
print("Trans. Ki67", H_ki67)


# %%
def reject_outliers(data, m=3):
    dt = data.copy()
    dt[(abs(data - np.mean(data)) > m * np.std(data))] = np.mean(data)
    return dt


def in_range(d):
    return (0, np.max(cv.GaussianBlur(d.copy(), (3, 3), 0)))


def flip8(imname):
    imSrc = cv.imread(imname)
    for kr in range(4):
        for pr in range(2):
            if kr + pr == 0: continue
            im = imSrc
            for k in range(kr):
                im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
            if pr != 0:
                im = cv.flip(im, flipCode=1)
            cv.imwrite(imname[:-4] + "[r%dp%d].tif" % (kr * 90, pr), im)
    return 0


def norm_by_row(M):
    for k in range(M.shape[1]):
        M[k, :] /= np.sqrt(np.sum(np.power(M[k, :], 2)))
    return M


def showbychan(im_ihc):
    for k in range(3):
        plt.figure()
        plt.imshow(im_ihc[:, :, k], cmap="gray")


def rgbdeconv(rgb, conv_matrix, C=0):
    rgb = rgb.copy().astype(float)
    rgb += C
    stains = np.reshape(-np.log10(rgb), (-1, 3)) @ conv_matrix
    return np.reshape(stains, rgb.shape)


def hecconv(stains, conv_matrix, C=0):
    #     from skimage.exposure import rescale_intensity
    stains = stains.astype(float)
    logrgb2 = -np.reshape(stains, (-1, 3)) @ conv_matrix
    rgb2 = np.power(10, logrgb2)
    return np.reshape(rgb2 - C, stains.shape)


def surf(matIn, name="fig", div=(50, 50), SIZE=(8, 6)):
    x = np.arange(0, matIn.shape[0])
    y = np.arange(0, matIn.shape[1])
    x, y = np.meshgrid(y, x)
    fig = plt.figure(figsize=SIZE)
    ax = Axes3D(fig)
    ax.plot_surface(x, y, matIn, rstride=div[0], cstride=div[1], cmap='jet')
    plt.title(name)
    plt.show()


def heunmix(im_hex):
    H = H_he
    Hinv = linalg.inv(norm_by_row(H))

    img = cv.cvtColor(im_hex, cv.COLOR_BGR2RGB) / 255.
    img[img == 0] = 1E-9
    im_sepa_hex = abs(rgbdeconv(img, Hinv))
    h = im_sepa_hex[:, :, 0];
    e = im_sepa_hex[:, :, 1];
    d_r = im_sepa_hex[:, :, 2];

    # h1 = rescale_intensity(h, in_range=in_range(reject_outliers(h)))
    # e1 = rescale_intensity(h, in_range=in_range(reject_outliers(e)))

    lbl = np.dstack([e, d_r, h])
    surf(h, "H")
    surf(e, "e")

    return lbl


def ihcunmix(im_ki67):
    # ## 实验：ki67染色解组及重组

    H = H_ki67
    Hinv = linalg.inv(norm_by_row(H))

    img = cv.cvtColor(im_ki67, cv.COLOR_BGR2RGB) / 255

    img[img == 0] = 1E-9
    im_sepa_ki67 = abs(rgbdeconv(img, Hinv))

    h = im_sepa_ki67[:, :, 0];
    e_r = im_sepa_ki67[:, :, 1];
    d = im_sepa_ki67[:, :, 2];

    # h1 = rescale_intensity(h, in_range=in_range(reject_outliers(h)))
    # d1 = rescale_intensity(d, in_range=in_range(reject_outliers(d)))
    #
    # d2 = (d1 * 255).astype(uint8)
    #
    # guide = d2
    #
    # geps = 1000;
    # grad = 10
    # guidedDAB = ximgproc.guidedFilter(guide=guide, src=d2, radius=grad, eps=geps, dDepth=-1)
    # gd = ximgproc.guidedFilter(guide=guide, src=d2, radius=grad, eps=geps, dDepth=-1)
    # gd = rescale_intensity(gd, in_range=in_range(gd))
    #
    # _, dmask = cv.threshold((guidedDAB * 255).astype(uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #
    # guidedDAB = ((255 - dmask) > 0).astype(uint8)
    # # plt.figure(figsize=(20,10));plt.subplot(121);plt.imshow(guidedDAB, cmap = "gray");plt.axis('off')
    #
    # MOP_SIZE, MCL_SIZE = 5, 5
    # guidedDAB = cv.morphologyEx(guidedDAB, op=cv.MORPH_OPEN, kernel=np.ones((MOP_SIZE, MOP_SIZE), uint8))
    # guidedDAB = cv.morphologyEx(guidedDAB, op=cv.MORPH_CLOSE, kernel=np.ones((MCL_SIZE, MCL_SIZE), uint8))
    #
    # # plt.subplot(122);plt.imshow(guidedDAB, cmap = "gray");plt.axis('off')
    #
    # h1 = rescale_intensity(h,
    #                        in_range=in_range(reject_outliers(h)))
    # d1 = rescale_intensity(d, in_range(reject_outliers(d)))
    #
    # lbl = np.dstack([h1, guidedDAB * d1])
    # lbl = np.dstack([lbl, zeros_like(h1)])
    lbl = np.dstack([e_r, d, h])
    plt.figure(figsize=(10, 10));
    plt.imshow(lbl);
    plt.show()
    return lbl


im_ki67 = cv.imread("./datasets/ihc_4d/trainA/18_10.tif", cv.CV_32F)
im_HE = cv.imread("./datasets/ihc_4d/trainB/18_10.tif", cv.CV_32F)
ihcunmix(im_ki67)
heunmix(im_HE)
