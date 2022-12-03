from keras_preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow as tf
import os

def get_gray_images(directory:str, size=-1, imgTotal = 1000): #-> list[tf.Tensor]:
    '''
    inputs - directory where the files to be converted are stored

    returns - list of tf.Tensors which represent the images in the directory that have been grayscaled
    '''
    ### Load get the files names from all folder
    training_file_names = []
    i = 0
    with os.scandir(directory) as entries:  
        for entry in entries:
            if i > imgTotal: break # imgTotal restraint    
            else:   
                training_file_names.append(entry.name)
                i+=1
    ### Pull all the images in and put them in an array
    image_array = []
    for img in training_file_names:
        image_array.append(img_to_array(load_img(f'{directory}/{img}')))
    ### Grayscale Images
    grayscale_imgs = []
    for rgb_img in image_array:
        tensor = tf.image.rgb_to_grayscale(rgb_img, name=None)
        if (size > 0):
            tensor = tf.image.resize(tensor, [size, size])
        grayscale_imgs.append(tensor)
    
    return grayscale_imgs

def get_hsv_images(directory:str, size=-1, imgTotal = 1000): #-> list[tf.Tensor]:
    '''
    HSV images bifurcate image intensity and color data- they are usually considered less noisy in the training process.
    inputs - directory where the files to be converted are stored

    returns - list of tf.Tensors which represent the images in the directory that have been converted to hsv
    '''
    ### Load get the files names from all folder
    with tf.device('/CPU:0'):
        training_file_names = []
        i = 0
        with os.scandir(directory) as entries:
            for entry in entries:
                if i > imgTotal: break     
                else:   
                    training_file_names.append(entry.name)
                    i+=1
        ### Pull all the images in and put them in an array
        image_array = []
        for img in training_file_names:
            image_array.append(img_to_array(load_img(f'{directory}/{img}')))
        ### HSV Images
        hsv_imgs = []
        for rgb_img in image_array:
            tensor = tf.image.rgb_to_hsv(rgb_img, name=None)
            if (size > 0):
                tensor = tf.image.resize(tensor, [size, size])
            hsv_imgs.append(tensor)
        
        return hsv_imgs

def get_saturated_images(directory:str, size=-1, imgTotal = 1000): #-> list[tf.Tensor]:
    '''
    inputs - directory where the files to be converted are stored

    returns - list of tf.Tensors which represent the images in the directory that have had their saturation increased
    '''
    ### Load get the files names from all folder
    training_file_names = []
    i = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if i > imgTotal: break     
            else:   
                training_file_names.append(entry.name)
                i+=1
    ### Pull all the images in and put them in an array
    image_array = []
    for img in training_file_names:
        image_array.append(img_to_array(load_img(f'{directory}/{img}')))
    ### Saturated Images
    saturated_imgs = []
    for rgb_img in image_array:
        tensor = tf.image.adjust_saturation(rgb_img, 1) # use contrast factor of 0.5
        if (size > 0):
            tensor = tf.image.resize(tensor, [size, size])
        saturated_imgs.append(tensor)
    
    return saturated_imgs
