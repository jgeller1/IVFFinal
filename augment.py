import random
import os
import numpy as np
""" 
File to augment blastoyst data
"""

from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from skimage import exposure
from PIL import Image 


def random_rotation(image_array):
    """ 
    Adds in a random rotation to the data between -360 and 360 degrees

    Parameters: 

    image_array (ndarray)

    Returns:
    
    transformed ndarray

    """
    random_degree = random.uniform(-360, 360)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array):
    """ 
    Adds in a random noise to the picture

    Parameters: 

    image_array (ndarray)

    Returns:
    
    transformed ndarray

    """
    return sk.util.random_noise(image_array)

def scale_in(image_array):
    """ 
    Scales the data down. Does not resize (the model handles this)

    Parameters: 

    image_array (ndarray)

    Returns:
    
    transformed ndarray

    """
    return sk.transform.rescale(image_array, scale=0.5, mode='constant')

def scale_out(image_array):
    """ 
    Scales the data down. Does not resize (the model handles this)

    Parameters: 

    image_array (ndarray)

    Returns:
    
    transformed ndarray

    """
    return sk.transform.rescale(image_array, scale=2, mode='constant')

def horiz_flip(image_array):
    """ 
    Horizontally flips image

    Parameters: 

    image_array (ndarray)

    Returns:
    
    transformed ndarray

    """
    return image_array[:, ::-1]


def vert_flip(image_array):
    """ 
    Vertically flips image

    Parameters: 

    image_array (ndarray)

    Returns:
    
    transformed ndarray

    """
    return image_array[::-1, :]

def contrast(image_array):
    """ 
    Randomly augments the contrast of the image

    Parameters: 

    image_array (ndarray)

    Returns:
    
    transformed ndarray

    """
    v_min, v_max = np.percentile(image_array, (0.2, 99.8))
    better_contrast = exposure.rescale_intensity(image_array, in_range=(v_min, v_max))
    return better_contrast


def brightness(image_array):
    """ 
    Adjusts brightness of image

    Parameters: 

    image_array (ndarray)

    Returns:
    
    transformed ndarray

    """
    gamma = np.random.uniform(low=.5, high=1.5, size=1)
    return exposure.adjust_gamma(image_array, gamma=gamma,gain=1)

def hue(image_array):
    """ 
    Adds random hue tint to image

    Parameters: 

    image_array (ndarray)

    Returns:
    
    transformed ndarray

    """
    f0, f1, f2 = np.random.uniform(low=.47, high=.63, size=3)

    r = image_array[:, :, 0]
    g = image_array[:, :, 1]
    b = image_array[:, :, 2]
    zeros = np.zeros(r.shape)
    rimg = np.stack((r*f0, g*f1, b*f2), axis=2)
    return rimg

if __name__ == '__main__':
    #Train set images path
    folder_path = 'Images_TwoClasses/train/NotPregnant'
    
    available_transformations = {
    'hue':hue,
    'brightness':brightness,
    'contrast':contrast,
    'vert_flip':vert_flip,
    'horiz_flip':horiz_flip,
    #'scale_out':scale_out,
    #'scale_in':scale_in,
    'random_noise':random_noise,
    'random_rotation':random_rotation
    }

    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f != '.DS_Store']

    #Number of new images to generate 
    num_files_desired = len(images)*1000
    num_generated_files = 0
    while num_generated_files < num_files_desired:
        print("iteration: {}".format(num_generated_files))
        #random image from the folder
        image_path = np.random.choice(images,replace=True)
        #read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)
        print(image_path)

        #random num of transformations to apply. Uses a geometric distribution with
        #a mean of 2 transformations per image 
        num_transformations_to_apply = np.minimum(int(np.random.geometric(p=.5, size=1)),len(available_transformations))
        print(num_transformations_to_apply)
        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1

        # define a name for our new file
        new_file_path = image_path[:-4] + '_augmented_' + str(num_generated_files) + '.jpg'

        # write image to the disk
        sk.io.imsave(new_file_path, transformed_image)  
        num_generated_files += 1
