"""
this code is for input a data from given image folder
it will pre process these images for nn models (VGG model)
return them in dictionary on their ids
"""
#TODO: add image pre processing support for multiple model networks

from os import listdir
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

def load_image_data(dir_name):
    images_dict = dict()
    for name in listdir(dir_name):
        file_name = dir_name + '/' + name
        img = load_img(file_name, target_size=(224, 224)) #TODO:change hardcode
        img = img_to_array(img)
        img_reshape = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        image = preprocess_input(img_reshape)
        img_id = name.split('.')[0]
        images_dict[img_id] = image
    return images_dict
# test code
dir_name = "data/Flicker8k_Dataset"
test_img = load_image_data(dir_name)
print(len(test_img))