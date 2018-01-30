from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input


# extract features from each photo in the given image directory
def extract_features(img_dir):
    input_layer = Input(shape=(224, 224, 3))
    model = VGG16(include_top=False, input_tensor=input_layer)
    print(model.summary())
    # extract features from each photo
    features = dict()
    for name in listdir(img_dir):
        filename = img_dir + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
        print('>%s' % name)
    return features


# extract features from all images
img_dir = 'Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))
