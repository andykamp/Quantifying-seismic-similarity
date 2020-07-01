# now we perform augmentation 
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from skimage import exposure
from  random import random
# Basic data augmentation techniques:
# Flipping: flipping the image vertically or horizontally
# Rotation: rotates the image by a specified degree.
# Shearing: shifts one part of the image like a parallelogram
# translation: 
# scale: 
# Cropping: object appear in different positions in different proportions in the image
# Changing brightness or contrast

def rot(img, rot_x=-150, rot_y=30):
    # rotation
    rotate=iaa.Affine(rotate=(rot_x, rot_y))
    rotated_image=rotate.augment_image(img)
    return rotated_image

def noise(img, noise_x=1, noise_y=2):
    # adding noise 
    gaussian_noise=iaa.AdditiveGaussianNoise(noise_x, noise_y)
    noise_image=gaussian_noise.augment_image(img)
    return noise_image

def crop(img, crop_x=0, crop_y=0.3):
    # cropping and resiszing the image 
    crop = iaa.Crop(percent=(crop_x, crop_y)) # crop image
    corp_image=crop.augment_image(img)
    return corp_image

def zom_indp(img, zoom_x=(1.5,1.0), zoom_y=(1.5,1.0)):
    # zoom axis indepentently 
    scale_im=iaa.Affine(scale={"x": zoom_x, "y": zoom_y})
    scale_image =scale_im.augment_image(img)
    return scale_image

def shear(img, shear_x=0, shear_y=40):
    # Shearing the image by 0 to 40 degrees
    shear = iaa.Affine(shear=(shear_x, shear_y))
    shear_image=shear.augment_image(img)
    return shear_image

def flip_hor(img, p=1.):
    #flipping image horizontally
    flip_hr=iaa.Fliplr(p=p)
    flip_hr_image= flip_hr.augment_image(img)
    return flip_hr_image


def flip_vert(img, p=1.):
    # flippping image vertically 
    flip_vr=iaa.Flipud(p=p)
    flip_vr_image= flip_vr.augment_image(img)
    return flip_vr_image

def bright(img, gamma=2.0):

    # changing the brightness of the image 
    # need to rescale intensitites in the image to do so 
    rescaled_img = exposure.rescale_intensity(img,in_range=(0, 1))
    contrast=iaa.GammaContrast(gamma=gamma)
    contrast_image =contrast.augment_image(rescaled_img)
    return contrast_image

def augment(img):
    if random() > 0.5:

        img = rot(img)
    if random() > 0.5:
       img =  noise(img)
    if random() > 0.5:
       img =  noise(img)
    if random() > 0.5:
       img = crop(img)
    if random() > 0.5:
       img =  zom_indp(img)
    if random() > 0.5:
       img =  shear(img)
    if random() > 0.5:
       img =  flip_hor(img)
    if random() > 0.5:
       img =  flip_vert(img)
    return img
        