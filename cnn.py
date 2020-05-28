import tensorflow as tf

class BaseConvolutionalNetwork(object):
    hyperparam = {
        'batch_size'    : 20, 
        'epoch'         : 100,
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
    
    def __init__(self, input_, load=False, mode='binary'):
        if not load:
            self.create_new()
        else:
            self.load(load)
        self.input_shape = input_
        self.mode = mode
            
    def show_summary(self):
        self.model.summary()

    def load(self, filename, obj='model'):
        if obj=='model':
            self.model = tf.keras.models.load_model(filename)
        elif obj=='weights_only':
            self.model.load_weights(filename)

    def load(self, filename, obj='model'):
        if obj=='model':
            self.model = tf.keras.models.load_model(filename)
        elif obj=='weights_only':
            self.model.load_weights(filename)
        else:
            print('Load failed. Unknown mode')

    def save(self, filename, obj='model'):
        path = 'saved\\'+filename
        if obj=='model':
            self.model.save(path)
        elif obj=='weights_only':
            self.model.save_weights(path+'\\'+variables)
        else:
            print('Load failed. Unknown mode')        

    def reset_model(self, mode='metrics_only'):
        if mode=='model':
            self.create_new()
        elif mode=='metrics_only':
            self.model.reset_metrics()
        else:
            print('Load failed. Unknown mode')

    def create_new(self):
        self.model = tf.keras.models.Sequential()        

    def set_data_dir(self, train_dir, val_dir):
        self.data_dir = {'train': train_dir, 'validation': val_dir}        
    
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
        self.model.add(tf.keras.layers.Flatten())
        
    def set_hidden_layers(self, neurons_list, activation='relu'):
        for num in neurons_list:
            hidden = tf.keras.layers.Dense(num, activation=activation)
            self.model.add(hidden)

    def set_output_layer(self, out_neurons, activation='sigmoid'):
        if self.mode == 'binary':
            activation = 'sigmoid'
        elif self.mode == 'categorical':
            activation = 'softmax'
        output = tf.keras.layers.Dense(out_neurons, activation=activation)
        self.model.add(output)

    def set_parameter(self, param, value):
        self.hyperparam[param] = value

    def set_augmentation(self, param, value):
        self.augmentation[param] = value
        
    def set_augmentations(self, aug_dict):
        for key in aug_dict:
            self.set_augmentation(key, aug_dict[key])

    def flow_from_directory(self):
        IMG = tf.keras.preprocessing.image.ImageDataGenerator
        # tg = IMG(**self.augmentation)
        tg = IMG(rescale = 1.0/255.)
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
            target_size = self.input_shape
        )
        return train_gen, validation_gen

    def set_callbacks(self, checkpoint):
        callbacks = []

        CB = tf.keras.callbacks
        impatient = CB.EarlyStopping(monitor='accuracy', patience=3)
        callbacks.append(impatient)
        if checkpoint:
            path = "saved\\"+checkpoint
            checkpoint_save = CB.ModelCheckpoint(filepath=path)
            callbacks.append(checkpoint_save)
        return callbacks

    def train(self, checkpoint=None):
        p = self.hyperparam
        train_generator, validation_generator = self.flow_from_directory()
        opt = tf.keras.optimizers.RMSprop(lr=p['learning_rate'])
        loss = self.mode + '_crossentropy'

        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        callbacks = self.set_callbacks(checkpoint)
        history = self.model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = p['steps'],
            epochs = p['epoch'],
            validation_steps = p['val_steps'],
            verbose = 2,
            callbacks = callbacks
        )
        return history
    
    def check_performance(self, paths=[]):
        # predicting images
        
        img = image.load_img(path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(fn)
        print(classes)        
        
     
#   # predicting images
#   path='/content/' + fn
#   img=image.load_img(path, target_size=(150, 150))
  
#   x=image.img_to_array(img)
#   x=np.expand_dims(x, axis=0)
#   images = np.vstack([x])
  
#   classes = model.predict(images, batch_size=10)
  
#   print(classes[0])
  
#   if classes[0]>0:
#     print(fn + " is a dog")
    
#   else:
#     print(fn + " is a cat")
