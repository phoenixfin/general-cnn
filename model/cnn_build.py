import tensorflow as tf
from .cnn import BaseConvolutionalNetwork

class SelfDefinedNetwork(BaseConvolutionalNetwork):
    def __init__(self, name_dir, save_file=''):
        super().__init__(name_dir, save_file)
        self.model = self.create_new()    
        
    def create_new(self):
        self.model = tf.keras.models.Sequential()        

    def _add_convolution(self, filter_num, kernels_size, 
                        pooling = (2,2), activation = 'relu', 
                        dropout = 0, normalize = False, first = True):
        kwargs = {'activation':activation}
        in_ = self.hyperparam['input_size']
        for i, num in enumerate(filter_num):
            kwargs['filters'] = num
            kwargs['kernel_size'] = kernels_size[i]
            if first and i==0: 
                kwargs['input_shape']= (in_, in_, 3)

            self.model.add(tf.keras.layers.Conv2D(**kwargs))

            if pooling: 
                self.model.add(tf.keras.layers.MaxPooling2D(*pooling))
            if normalize:
                self.model.add(tf.keras.layers.BatchNormalization())

        if dropout:
            self.model.add(tf.keras.layers.Dropout(dropout))

        self.model.add(tf.keras.layers.Flatten())
        
    def _add_hidden_layers(self, neurons_list, dropout = 0, 
                          normalize = False, activation='relu'):
        for num in neurons_list:
            hidden = tf.keras.layers.Dense(num, activation=activation)           
            self.model.add(hidden)
            if dropout:
                self.model.add(tf.keras.layers.Dropout(dropout))
            if normalize:
                self.model.add(tf.keras.layers.BatchNormalization())
                
    def _set_output_layer(self, out_neurons, activation):
        output = tf.keras.layers.Dense(out_neurons, activation=activation)
        self.model.add(output)                