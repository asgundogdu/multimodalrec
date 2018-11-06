import os.path, re
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
from tqdm import tqdm
import numpy as np


class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        # if self.weights is None:
        #     # For imagenet/default network:
        #     features = features[0]
        # else:
        #     # For loaded network:
        features = features[0]

        return features


def extract_features(_dir_=''):
    seq_lenght = 30
    train_dir = [_dir_+'train/'+f for f in os.listdir(_dir_+'train/') if f.find('.')==-1]
    test_dir = [_dir_+'test/'+f for f in os.listdir(_dir_+'test/') if f.find('.')==-1]
    # print(test_dir)
    all_data = train_dir+test_dir

    model = Extractor()

    pbar = tqdm(total=len(all_data))
    for trailer_dir in all_data:
        # Get the path to the frames for this trailer
        trailer_dir

        # Loop through and extract features to build the sequence.
        frames = [trailer_dir+'/'+f for f in os.listdir(trailer_dir) if bool(re.search("[0-9].jpg", f))]
        # print(frames)
        sequence = []
        for image in frames:
            features = model.extract(image)
            sequence.append(features)

        np.save('sequences/'+trailer_dir.rsplit('/')[-1]+'.seq', sequence)
        pbar.update(1)
    pbar.close()



# def read_process_image(im_directory):
#     img_path = im_directory
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     return x


# def image_data_creation(directory='sampled_images/'):


# def ImageEncoder(Visual_data, extraction_type, pretrained_model):    
#     if pretrained_model == 'MobileNetV2':
#         model = MobileNetV2(weights='imagenet', include_top=False)
#         x = preprocess_input(frame)
#         features = model.predict(x)
#         return features