import tensorflow as tf
from .cnn import BaseConvolutionalNetwork

class TransferNetwork(BaseConvolutionalNetwork):

    def __init__(self, name_dir, save_file=''):
        super().__init__(name_dir, save_file)

    def set_pretrained_model(self, model_name='InceptionV3', 
                             lastpool='avg', fine_tune_at=None,
                             feature_only=True):
        size = self.hyperparam['input_size']
        self.premodel = getattr(tf.keras.applications, model_name)(
            include_top=not feature_only, 
            input_shape=(size, size, 3),
            pooling=lastpool, 
        )
        self.model = self.premodel
        self.last_output = self.premodel.layers[-1].output        
        if lastpool == None:
            self.last_output = tf.keras.layers.Flatten()(self.last_output) 
        
        self.set_fine_tuning(fine_tune_at)

        
    def rebase_model(self):
        self.model = self.premodel
        
    def set_last_layer(self, ids=None, mode='name'):
        last_layer = self.premodel.get_layer(**{mode: ids})
        self.last_output = last_layer.output

    def set_fine_tuning(self, fine_tune_at):
        if type(fine_tune_at) == int:
            fine_tune_at = self.premodel.get_layer(index=fine_tune_at).name
        val = False
        count = 0

        for layer in self.premodel.layers:
            if fine_tune_at:
                if layer.name == fine_tune_at: 
                    val = True
            count += int(val)
            layer.trainable = val
        print('trainable layers:', count)
        print('total layers:', len(self.premodel.layers))

    def _add_convolution(self, filter_num, kernels_size, 
                        pooling = (2,2), activation = 'relu', 
                        dropout = 0, normalize = False):

        kwargs = {'activation':activation}
        for i, num in enumerate(filter_num):
            kwargs['filters'] = num
            kwargs['kernel_size'] = kernels_size[i]

            x = tf.keras.layers.Conv2D(**kwargs)(self.last_output)
            if pooling: 
                x = tf.keras.layers.MaxPooling2D(*pooling)(x)
            if normalize:
                x = tf.keras.layers.BatchNormalization()(x)

        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Flatten()(x)
        self.last_output = x
        self.model = tf.keras.models.Model(self.model.input, x)
        
    def _add_hidden_layers(self, neurons_list, dropout = 0, 
                          normalize = False, activation='relu'):
        x = self.last_output
        for num in neurons_list:
            x = tf.keras.layers.Dense(num, activation=activation)(x)
            if dropout:
                x = tf.keras.layers.Dropout(dropout)(x)
            if normalize:
                x = tf.keras.layers.BatchNormalization()(x)        
        self.last_output = x
        self.model = tf.keras.models.Model(self.model.input, x)        
        
    def _set_output_layer(self, out_neurons, activation):
        out_layer = tf.keras.layers.Dense(out_neurons, activation=activation)
        x = out_layer(self.last_output)

        self.model = tf.keras.models.Model(self.model.input, x)