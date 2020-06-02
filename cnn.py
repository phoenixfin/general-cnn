import tensorflow as tf
import numpy as np
import os, random, logging
from helper.report import plot_accuracy

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

class BaseConvolutionalNetwork(object):
    callbacks_list = [
        'no_progresss_stopping', 
        'save_per_epoch', 
        'log_to_csv', 
        'good_result_stopping'
    ]    
    
    # Default Hyperparameter
    hyperparam = {
        'batch_size'    : 20, 
        'epoch'         : 100,
        'steps'         : 100,
        'val_steps'     : 50,
        'learning_rate' : 0.001            
    }
    
    # Default Augmentation
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
    
    def __init__(self, name_dir='', input_=(), load=False, mode='binary'):
        if not load:
            self.create_new()
        else:
            self.load(load)
        self.input_shape = input_
        self.mode = mode
        self.name = name_dir
        self.obtain_image_data()
            
    def show_summary(self):
        self.model.summary()

    def load(self,obj='model'):
        file = 'saved\\'+self.name
        if obj=='model':
            self.model = tf.keras.models.load_model(file)
        elif obj=='weights_only':
            self.model.load_weights(file+'\\variables\\')
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

    def set_output_layer(self, activation='sigmoid'):
        if self.mode == 'binary':
            activation = 'sigmoid'
        elif self.mode == 'categorical':
            activation = 'softmax'
        out_neurons = self.image_data['train'].num_classes
        output = tf.keras.layers.Dense(out_neurons, activation=activation)
        self.model.add(output)

    def set_parameter(self, param, value):
        self.hyperparam[param] = value

    def set_augmentation(self, param, value):
        self.augmentation[param] = value
        
    def set_augmentations(self, aug_dict):
        for key in aug_dict:
            self.set_augmentation(key, aug_dict[key])

    def obtain_image_data(self):
        base_dir = os.path.join(os.getcwd(), 'data', self.name)
        train_dir = base_dir+'\\tobeused\\train'
        val_dir = base_dir+'\\tobeused\\validation'
                
        IMG = tf.keras.preprocessing.image.ImageDataGenerator
        # tg = IMG(**self.augmentation)
        tg = IMG(rescale = 1.0/255.)
        vg = IMG(rescale = 1.0/255.)

        train_data = tg.flow_from_directory(
            train_dir,
            batch_size = self.hyperparam['batch_size'],
            class_mode = self.mode,
            target_size = self.input_shape
        )
        validation_data =  vg.flow_from_directory(
            val_dir,
            batch_size = self.hyperparam['batch_size'],
            class_mode = self.mode,
            target_size = self.input_shape
        )
        self.image_data = {'train':train_data, 'val':validation_data}

    def set_callbacks(self, cb_list):
        callbacks = []
        CB = tf.keras.callbacks

        # no progress stopping callback
        if 'no_progress_stopping' in cb_list:
            impatient = CB.EarlyStopping(monitor='accuracy', patience=3)
            callbacks.append(impatient)

        # save per epoch callback
        if 'save_per_epoch' in cb_list:
            checkpoint_save = CB.ModelCheckpoint(
                filepath="saved\\"+self.name, 
                save_best_only=True)
            callbacks.append(checkpoint_save)

        # log to csv file callback
        if 'log_to_csv' in cb_list:
            logger = CB.CSVLogger('log\\'+self.name)
            callbacks.append(logger)

        # stop when enough callback
        if 'good_result_stopping' in cb_list:
            def stopper(epoch, logs):
                if logs['accuracy']>0.97: self.model.stop_training = True
            good_res = CB.LambdaCallback(on_epoch_end=lambda e,l: stopper(e,l))
            callbacks.append(good_res)
                            
        return callbacks

    def train(self, save=False):
        p = self.hyperparam
        
        # Compile the model
        opt = tf.keras.optimizers.RMSprop(lr=p['learning_rate'])
        loss = self.mode + '_crossentropy'
        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

        # Set some callbacks
        cb_list = self.callbacks_list.copy()
        if not save:
            cb_list.remove('save_per_epoch')
        callbacks = self.set_callbacks(cb_list)

        # Fitting the data!
        history = self.model.fit(
            self.image_data['train'],
            validation_data = self.image_data['val'],
            steps_per_epoch = p['steps'],
            epochs = p['epoch'],
            validation_steps = p['val_steps'],
            verbose = 2,
            callbacks = callbacks
        )
        return history
     
    def predict(self, num):
        IM = tf.keras.preprocessing.image

        data_dir = os.path.join(os.getcwd(), 'data', self.name)
        category = os.listdir(data_dir+'\\tobeused\\train')
        path = os.path.join(data_dir+'\\test') 
        test_data = os.listdir(path)
        randomized = random.sample(test_data, num)

        for file in randomized:
            img = IM.load_img(path+'\\'+file, target_size = self.input_shape)

            x = IM.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            res = self.model.predict(images, batch_size=10)

            print(file, ': ',category[np.argmax(res[0])])

    def report(self, history):
        plot_accuracy(history)