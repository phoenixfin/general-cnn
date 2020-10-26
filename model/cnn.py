import tensorflow as tf
import numpy as np
import os
import random
from helper.report import plot_from_history
from helper import constants

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())


class BaseConvolutionalNetwork(object):
    callbacks_list = [
        'no_progresss_stopping',
        'save_per_epoch',
        'log_to_csv',
        'good_result_stopping'
    ]

    def __init__(self, name_dir, save_file=''):
        self.model = None
        self.name = name_dir
        self.hyperparam = constants.hyperparameter
        self.augmentation = constants.augmentation
        self._obtain_image_data()
        self.save = save_file if save_file else name_dir

    def show_summary(self):
        self.model.summary()

    def load(self, obj='model', path=None):
        if path == None:
            path = self.save
        file = 'saved\\'+path
        if obj == 'model':
            self.model = tf.keras.models.load_model(file)
        elif obj == 'weights_only':
            self.model.load_weights(file+'\\variables\\')
        else:
            print('Load failed. Unknown mode')

    def save(self, filename, obj='model'):
        path = 'saved\\'+filename
        if obj == 'model':
            self.model.save(path)
        elif obj == 'weights_only':
            self.model.save_weights(path+'\\'+variables)
        else:
            print('Load failed. Unknown mode')

    def reset_model(self, mode='metrics_only'):
        if mode == 'model':
            self.create_new()
        elif mode == 'metrics_only':
            self.model.reset_metrics()
        else:
            print('Load failed. Unknown mode')

    def add_convolution(self, **kwargs):
        self._add_convolution(**kwargs)

    def add_hidden_layers(self, **kwargs):
        self._add_hidden_layers(**kwargs)

    def set_output_layer(self):
        if self.mode == 'binary':
            activation = 'sigmoid'
        elif self.mode == 'categorical':
            activation = 'softmax'
        out_neurons = self.image_data['train'].num_classes
        self._set_output_layer(out_neurons, activation)

    def _obtain_image_data(self):
        base_dir = os.path.join(os.getcwd(), 'data', self.name)
        train_dir = base_dir+'\\tobeused\\train'
        val_dir = base_dir+'\\tobeused\\validation'

        IMG = tf.keras.preprocessing.image.ImageDataGenerator

        # tg = IMG(**self.augmentation)
        tg = IMG(rescale=1./255,
                 zoom_range=0.2,
                 horizontal_flip=True,
                 width_shift_range=0.2,
                 height_shift_range=0.2,
                 shear_range=0.2)
        vg = IMG(rescale=1./255)

        in_ = self.hyperparam['input_size']
        classes = len(os.listdir(train_dir))
        self.mode = 'binary' if classes == 2 else 'categorical'

        train_data = tg.flow_from_directory(
            train_dir,
            batch_size=self.hyperparam['batch_size'],
            class_mode=self.mode,
            target_size=(in_, in_)
        )
        validation_data = vg.flow_from_directory(
            val_dir,
            batch_size=self.hyperparam['batch_size'],
            class_mode=self.mode,
            target_size=(in_, in_)
        )
        self.image_data = {'train': train_data, 'val': validation_data}

    def set_callbacks(self, cb_list):
        callbacks = []
        CB = tf.keras.callbacks

        # no progress stopping callback
        if 'no_progress_stopping' in cb_list:
            impatient = CB.EarlyStopping(
                monitor='val_accuracy',
                min_delta=0.05,
                patience=3)
            callbacks.append(impatient)

        # save per epoch callback
        if 'save_per_epoch' in cb_list:
            checkpoint_save = CB.ModelCheckpoint(
                filepath="saved\\"+self.save,
                save_best_only=True)
            callbacks.append(checkpoint_save)

        # log to csv file callback
        if 'log_to_csv' in cb_list:
            logger = CB.CSVLogger('log\\'+self.save)
            callbacks.append(logger)

        # stop when enough callback
        if 'good_result_stopping' in cb_list:
            def stopper(epoch, logs):
                if logs['accuracy'] > 0.97:
                    self.model.stop_training = True
            good_res = CB.LambdaCallback(
                on_epoch_end=lambda e, l: stopper(e, l))
            callbacks.append(good_res)

        return callbacks

    def train(self, save=False):
        # Compile the model
        lr = self.hyperparam['learning_rate']
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
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
            validation_data=self.image_data['val'],
            steps_per_epoch=self.hyperparam['steps'],
            epochs=self.hyperparam['epochs'],
            validation_steps=self.hyperparam['val_steps'],
            callbacks=callbacks
        )
        return history

    def predict_imagenet(self, num):
        IM = tf.keras.preprocessing.image

        data_dir = os.path.join(os.getcwd(), 'data', self.name)
        category = imagenet_labels
        path = os.path.join(data_dir+'\\test')
        test_data = random.sample(os.listdir(path), num)
        size = self.hyperparam['input_size']

        for file in test_data:
            img = IM.load_img(path+'/'+file, target_size=[224, 224])
            x = IM.img_to_array(img)
            x = tf.keras.applications.mobilenet.preprocess_input(
                x[tf.newaxis, ...])
            res = self.premodel(x)

            decoded = imagenet_labels[np.argsort(res)[0, ::-1][:10]+1]
            print(file, ': ', decoded)

    def predict(self, num):
        IM = tf.keras.preprocessing.image

        data_dir = os.path.join(os.getcwd(), 'data', self.name)
        category = list(self.image_data['train'].class_indices.keys())
        path = os.path.join(data_dir+'\\test')
        test_data = random.sample(os.listdir(path), num)
        size = self.hyperparam['input_size']

        for file in test_data:
            img = IM.load_img(path+'\\'+file, target_size=(size, size))
            img_array = IM.img_to_array(img)
            normalized = np.expand_dims(img_array, axis=0)/255

            res = self.model.predict(normalized, batch_size=10)
            print(file, ': ', category[np.argmax(res[0])])

    def report(self, history):
        plot_from_history(history)
