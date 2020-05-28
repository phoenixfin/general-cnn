import tensorflow as tf

class BaseConvolutionalNetwork(object):
    hyperparam = {
        'batch_size'    : 20, 
        'epoch'         : 15,
        'steps'         : 100,
        'val_steps'     : 50,
        'learning_rate' : 0.001            
    }
    
    augmentation = {
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
    }
    
    def __init__(self, input_, train_dir, val_dir, mode='binary'):
        self.model = tf.keras.models.Sequential()
        self.input_shape = input_
        self.mode = mode
        self.data_dir = {'train': train_dir, 'validation': val_dir}

    def show_summary(self):
        self.model.summary()

    def add_convolution(self, filter_num, kernels_size, 
                        pooling = (2,2), activation = 'relu', 
                        dropout = 0, first = True):
        kwargs = {'activation':activation}
        for i, num in enumerate(filter_num):
            kwargs['filters'] = num
            kwargs['kernel_size'] = kernels_size[i]
            if first and i==0: kwargs['input_shape']=self.input_shape + (3,)
            self.model.add(tf.keras.layers.Conv2D(**kwargs))
            if pooling: 
                self.model.add(tf.keras.layers.MaxPooling2D(*pooling))
        if dropout:
            self.model.add(tf.keras.layers.Dropout(dropout))

    def flatten(self):
        self.model.add(tf.keras.layers.Flatten())
        
    def set_hidden_layers(self, neurons_list, activation='relu'):
        for num in neurons_list:
            self.model.add(tf.keras.layers.Dense(num, activation=activation))

    def set_output_layer(self, out_neurons, activation='sigmoid'):
        self.model.add(tf.keras.layers.Dense(out_neurons, activation=activation))
        
    def set_parameter(self, param, value):
        self.hyperparam[param] = value

    def set_augmentation(self, param, value):
        self.augmentation[param] = value
        
    def set_augmentations(self, aug_dict):
        for key in aug_dict:
            self.set_augmentation(key, aug_dict[key])

    def flow_from_directory(self):
        IMG = tf.keras.preprocessing.image.ImageDataGenerator
        tg = IMG(**self.augmentation)
        vg = IMG(rescale = 1.0/255.)

        train_gen = tg.flow_from_directory(
            self.data_dir['train'],
            batch_size = self.hyperparam['batch_size'],
            class_mode = self.mode,
            target_size = self.input_shape
        )
        validation_gen =  vg.flow_from_directory(
            self.data_dir['validation'],
            batch_size = self.hyperparam['batch_size'],
            class_mode = self.mode,
            target_size = (150, 150)
        )
        return train_gen, validation_gen

    def train(self):
        p = self.hyperparam
        train_generator, validation_generator = self.flow_from_directory()
        opt = tf.keras.optimizers.RMSprop(lr=p['learning_rate'])
        loss = 'binary_crossentropy' if self.mode=='binary' else 'categorical_crossentropy'
        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        
        history = self.model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = p['steps'],
            epochs = p['epoch'],
            validation_steps = p['val_steps'],
            verbose = 2
        )