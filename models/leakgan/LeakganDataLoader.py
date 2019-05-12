import numpy as np
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras import backend as K

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

def load_image(file_name):
    """
    Load and preprocess an image
    """
    img = image.load_img(file_name, target_size=(224, 224))
    im = image.img_to_array(img)
    im = np.expand_dims(im, axis=0)

    im = preprocess_input(im)
    return im

def get_imagefeatures_vgg19(vgg_model, batch_size, image_files, feature_files):
    features = []
    for i in range(batch_size):
        try:
            feature.append(np.load(feature_files[i]))
        except:
            fc2 = vgg_model.predict(self.load_image(images[i]))
            reshaped = np.reshape(fc2, (4096))  
            features.append(reshaped)
    
    return features

class DataLoader():
    def __init__(self, batch_size, seq_length, end_token=0, dataset=None):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token
        self.dataset = dataset
        net = VGG19(weights='imagenet')
        self.vgg_model = Model(input= net.input, output= net.get_layer('fc2').output)

    def create_batches(self, data_file):
        self.token_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def next_data_batch(self, is_train=True):

        batch = self.dataset.next_batch()

        image_files, feature_files, sentences, masks, sent_lens = batch
        if (is_train):
            conv_features = get_imagefeatures_vgg19(vgg_model, this.batch_size, image_files, feature_files)
        else:
            conv_features = []

        return sentences, conv_features

    def reset_pointer(self):
        self.pointer = 0


class DisDataloader():
    def __init__(self, batch_size, seq_length, dataset):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.seq_length = seq_length
        self.dataset = dataset
        net = VGG19(weights='imagenet')
        self.vgg_model = Model(input= net.input, output= net.get_layer('fc2').output)

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        """
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    positive_examples.append(parse_line)
        """
        positive_examples = self.dataset.word_idxs
        image_files = self.dataset.image_files

        self.image_files = np.array(image_files + image_files)

        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.images_batches = np.split(self.image_files, self.num_batch, 0)
        
        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def next_data_batch(self):

        batch = self.dataset.next_batch()

        image_files, feature_files, sentences, masks, sent_lens = batch
        if (is_train):
            conv_features = get_imagefeatures_vgg19(vgg_model, this.batch_size, image_files, feature_files)
        else:
            conv_features = []

        return sentences, conv_features, sent_lens

    def reset_pointer(self):
        self.pointer = 0
