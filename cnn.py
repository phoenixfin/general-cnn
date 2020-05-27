import tensorflow as tf

class BaseConvolutionalNetwork(object):
    hyperparam = {
        'batch_size'    : 20, 
        'epoch'         : 15
        'steps'         : 100,
        'val_steps'     : 50,
        'learning_rate' : 0.001            
    }
    
    def __init__(self, input_, train_dir, val_dir, mode='binary'):
        self.model = tf.keras.models.Sequential()
        self.input_shape = input_
        self.mode = mode
        self.data_dir = {'train': train_dir, 'validation': val_dir}

    def summary(self):
        self.model.summary()

    def add_convolution(self, filter_num, kernel_size, pooling=(2,2), activation='relu', first=True):
        kwargs = {'activation':activation}
        for i, num in enumerate(conv_num):
            kwargs['filters'] = num
            kwargs['kernel_size'] = kernel_size[i]
            if first and i==0: kwargs['input_shape']=self.input_shape + (3,)
            self.model.add(tf.keras.layers.Conv2D(**kwargs))
            if pooling: 
                self.models.add(tf.keras.layers.MaxPooling2D(*pooling))

    def flatten(self):
        self.model.add(tf.keras.layers.Flatten())
        
    def set_hidden_layers(self, neurons_list, activation='relu'):
        for num in neurons_list:
            self.model.add(tf.keras.layers.Dense(num, activation=activation))

    def set_output_layer(self, out_neurons, activation='sigmoid'):
        self.model.add(tf.keras.layers.Dense(out_neurons, activation=activation))
        
    def set_parameter(self, param, value):
        self.hyperparam[param] = value

    def flow_from_directory(self):
        IMG = tf.keras.preprocessing.image.ImageDataGenerator
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
            target_size = (150, 150)
        )
        return train_gen, validation_gen

    def train():
        p = self.hyperparam
        train_generator, validation_generator = self.flow_from_directory
        opt = tf.keras.optimizers.RMSprop(lr=p['learning_rate'])
        loss = 'binary_crossentropy' if mode='binary' else 'categorical_crossentropy'
        self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        
        history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = p['steps'],
            epochs = p['epoch'],
            validation_steps = p['val_steps'],
            verbose = 2
        )