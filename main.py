# -*- coding: utf-8 -*-

from cnn import BaseConvolutionalNetwork
import os
here = os.getcwd()

basic_augmentation = {
    'rescale'             : 1./255,
    'rotation_range'      : 40,
    'width_shift_range'   : 0.2,
    'height_shift_range'  : 0.2,
    'shear_range'         : 0.2,
    'zoom_range'          : 0.2,
    'horizontal_flip'     : True,
}

def cats_dogs_case(save=None):
    base_dir = here+ '\\data\\dogs-vs-cats\\tobeused'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    input_shape = (150,150)
    
    cnn = BaseConvolutionalNetwork(input_shape)
    cnn.set_data_dir(train_dir, validation_dir)
    cnn.add_convolution([16,32,64], (3,3)*3)
    cnn.set_augmentations(basic_augmentation)
    cnn.set_hidden_layers([512])
    cnn.set_output_layer(1)
    cnn.show_summary()
    cnn.train(save)

if __name__ == '__main__':
    cats_dogs_case('catdog_model')
