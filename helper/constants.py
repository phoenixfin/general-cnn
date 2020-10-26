hyperparameter = {
    'batch_size': 32,
    'epochs': 5,
    'learning_rate': 0.01,
    'input_size': 224,
    'steps': 100,
    'val_steps': 50
}

""" Default Augmentation
'featurewise_center'              : False, 
'samplewise_center'               : False,
'featurewise_std_normalization'   : False, 
'samplewise_std_normalization'    : False,
'zca_whitening'                   : False, 
'zca_epsilon'                     : 1e-06, 
'rotation_range'                  : 0, 
'width_shift_range'               : 0.0,
'height_shift_range'              : 0.0, 
'brightness_range'                : None, 
'shear_range'                     : 0.0, 
'zoom_range'                      : 0.0,
'channel_shift_range'             : 0.0, 
'fill_mode'                       : 'nearest', 
'cval'                            : 0.0, 
'horizontal_flip'                 : False,
'vertical_flip'                   : False, 
'rescale'                         : None, 
'preprocessing_function'          : None,
'data_format'                     : None, 
'validation_split'                : 0.0, 
'dtype'                           : None        
"""

augmentation = {
    'rescale': 1./255,
    # 'rotation_range'      : 10,
    # 'width_shift_range'   : 0.2,
    # 'height_shift_range'  : 0.2,
    # 'shear_range'         : 0.2,
    # 'zoom_range'          : 0.2,
    # 'horizontal_flip'     : True,
}

pretrained_lst = [
    'DenseNet121',
    'DenseNet169',
    'DenseNet201',
    'InceptionResNetV2',
    'InceptionV3',
    'MobileNet',
    'MobileNetV2',
    'NASNetLarge',
    'NASNetMobile',
    'ResNet101',
    'ResNet101V2',
    'ResNet152',
    'ResNet152V2',
    'ResNet50',
    'ResNet50V2',
    'VGG16',
    'VGG19',
    'Xception'
]

optimizer_lst = [
    'Adadelta',
    'Adagrad',
    'Adam',
    'Adamax',
    'Ftrl',
    'Nadam',
    'RMSprop',
    'SGD'
]
