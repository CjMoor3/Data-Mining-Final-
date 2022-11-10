from keras_preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow as tf
import os

def get_gray_images(directory:str) -> list[tf.Tensor]:
    '''
    inputs - directory where the files to be converted are stored

    returns - list of tf.Tensors which represent the images in the directory that have been grayscaled
    '''
    ### Load get the files names from all folder
    training_file_names = []
    with os.scandir(directory) as entries:
        for entry in entries:
            training_file_names.append(entry.name)
    ### Pull all the images in and put them in an array
    image_array = []
    for img in training_file_names:
        image_array.append(img_to_array(load_img(f'{directory}/{img}')))
    ### Grayscale Images
    grayscale_imgs = []
    for rgb_img in image_array:
        grayscale_imgs.append(
            tf.image.rgb_to_grayscale(
                rgb_img, name=None
            )
        )
    
    return grayscale_imgs
