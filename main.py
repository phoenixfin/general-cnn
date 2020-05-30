# -*- coding: utf-8 -*-

from cnn import BaseConvolutionalNetwork
import os
data_dir = os.getcwd()+'\\data'

basic_augmentation = {
    'rescale'             : 1./255,
    'rotation_range'      : 40,
    'width_shift_range'   : 0.2,
    'height_shift_range'  : 0.2,
    'shear_range'         : 0.2,
    'zoom_range'          : 0.2,
    'horizontal_flip'     : True,
}

def cats_dogs_case(save=False):
    input_shape = (150,150)
    cnn = BaseConvolutionalNetwork('dogs-vs-cats', input_shape)
    cnn.add_convolution([16,32,64], [(3,3)]*3)
    cnn.set_augmentations(basic_augmentation)
    cnn.set_hidden_layers([512])
    cnn.set_output_layer(1)
    cnn.train(save)

def human_horse_case(save=False):
    input_shape = (300,300)    
    cnn = BaseConvolutionalNetwork('horse-or-human',input_shape)
    cnn.add_convolution([16,32,64,64,64], [(3,3)]*5)
    cnn.set_augmentations(basic_augmentation)
    cnn.set_hidden_layers([512])
    cnn.set_output_layer(1)
    cnn.train(save)

def rock_paper_scissor_case(save=False):
    input_shape = (300,300)
    
    cnn = BaseConvolutionalNetwork('rps', input_shape, mode='categorical')
    cnn.add_convolution([64,64,128,128], [(3,3)]*4, dropout=0.5)
    cnn.set_augmentations(basic_augmentation)
    cnn.set_hidden_layers([512])
    cnn.set_output_layer(3)
    cnn.train(save)
        

if __name__ == '__main__':
    # cats_dogs_case(save=True)
    # human_horse_case(save=True)
    rock_paper_scissor_case(save = True)