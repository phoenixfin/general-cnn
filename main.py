# -*- coding: utf-8 -*-
from model import TransferNetwork, SelfDefinedNetwork
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def cats_dogs_case(save=False):
    input_shape = (150, 150)
    cnn = BaseConvolutionalNetwork('dogs-vs-cats', input_shape)
    cnn.add_convolution([16, 32, 64], [(3, 3)]*3)
    cnn.set_augmentations(basic_augmentation)
    cnn.set_hidden_layers([512])
    cnn.set_output_layer(1)
    cnn.train(save)


def human_horse_case(save=False):
    input_shape = (300, 300)
    cnn = BaseConvolutionalNetwork('horse-or-human', input_shape)
    cnn.add_convolution([16, 32, 64, 64, 64], [(3, 3)]*5)
    cnn.set_augmentations(basic_augmentation)
    cnn.set_hidden_layers([512])
    cnn.set_output_layer(1)
    cnn.train(save)


def rock_paper_scissor_case(save=False):
    input_shape = (300, 300)

    cnn = BaseConvolutionalNetwork('rps', input_shape, mode='categorical')
    cnn.add_convolution([64, 64, 128, 128], [(3, 3)]*4, dropout=0.5)
    cnn.set_augmentations(basic_augmentation)
    cnn.set_hidden_layers([512])
    cnn.set_output_layer(3)
    cnn.train(save)


def plant_disease_case(save=False):
    cnn = SelfDefinedNetwork('plant-diseases')
    cnn.add_convolution([64, 64, 128, 128], [(3, 3)]*4, dropout=0.5)
    cnn.set_augmentations(basic_augmentation)
    cnn.set_hidden_layers([512], dropout=0.2)
    cnn.set_output_layer()
    cnn.train(save)


def plant_disease_pretrained(save=False):
    cnn = TransferNetwork(
        'plant-diseases', 'plant-diseases-densenet-with-aug2')
    cnn.set_pretrained_model('DenseNet201', feature_only=False)
    cnn.predict_imagenet(10)
    # cnn.add_hidden_layers(neurons_list=[512, 512], dropout=0.2)
    # cnn.set_output_layer()
    # cnn.load(path='plant-diseases-densenet-with-aug')
    # cnn.show_summary()


if __name__ == '__main__':
    # cats_dogs_case(save=True)
    # human_horse_case(save=True)
    # plant_disease_pretrained(save = True)

    # cnn = TransferNetwork('plant-diseases','plant-diseases-mobilenet-with-aug')
    # cnn = BaseConvolutionalNetwork('plant-diseases', (300,300), mode='categorical')
    # cnn.load()
    # cnn.train(save=True)
    # cnn.predict(20)

    from helper.report import plot_from_log
    plot_from_log('log/plant-diseases-densenetgit')
