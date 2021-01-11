import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Activation, Flatten, Dense, Reshape, Input
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from tensorflow.python.client import device_lib
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data(image_size=(256, 256), batch_size=32, data_folder="PetImages"):
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        validation_split=0.2
    )
    train_ds = datagen.flow_from_directory(data_folder, subset="training", class_mode=None, batch_size=batch_size,
                                           target_size=image_size, interpolation="nearest")
    val_ds = datagen.flow_from_directory(data_folder, subset="validation", class_mode=None, batch_size=batch_size,
                                         target_size=image_size, interpolation="nearest")

    return train_ds, val_ds


def plot_images(images):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.subplots(4, 4).flatten()
    for i in range(len(ax)):
        ax[i].imshow(images[i])


class AutoEncoder:

    def create_encoder(self):
        encoder_input = keras.Input(shape=self.input_shape,
                                    #batch_size=self.batch_size,
                                    name="original_img")
        x = layers.experimental.preprocessing.Rescaling(1./255)(encoder_input)

        for f in self.filters:
            x = Conv2D(f, (3, 3)    , 2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(self.chanDim)(x)

        self.volumeSize = keras.backend.int_shape(x)
        x = Flatten()(x)
        x = Dense(self.latent_dim)(x)
        encoder_output = x

        encoder = keras.Model(encoder_input, encoder_output, name="Encoder",)
        return encoder_input, encoder_output, encoder

    def create_decoder(self, eager_execution=False):
        decoder_input = keras.Input(shape=self.encoder_output.shape[1:],
                                    #batch_size=self.batch_size,
                                    name="encoded_img")
        x = decoder_input

        x = Dense(tf.reduce_prod(self.volumeSize[1:]))(x)
        x = Reshape((self.volumeSize[1], self.volumeSize[2], self.volumeSize[3]))(x)
        for f in self.filters[::-1]:
            x = Conv2DTranspose(f, (3, 3), 2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=self.chanDim)(x)

        x = Conv2DTranspose(self.input_shape[-1], (3, 3), padding="same")(x)
        x = Activation("sigmoid")(x)

        decoder_output = layers.experimental.preprocessing.Rescaling(255.)(x)

        decoder = keras.Model(decoder_input, decoder_output, name="Decoder")
        return decoder_input, decoder_output, decoder


    def create_autoencoder(self):
        auto_input = keras.Input(shape=self.input_shape,
                                 #batch_size=self.batch_size,
                                 name="AutoEncoder_Input")
        encoded = self.encoder(auto_input)
        auto_output = self.decoder(encoded)
        autoencoder = keras.Model(auto_input, auto_output, name="AutoEncoder")
        return auto_input, auto_output, autoencoder

    def __init__(self, input_shape=(256, 256, 3), batch_size=32, latent_dim=16, fuckyou=(32, 64)):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.filters = fuckyou
        self.chanDim = -1
        self.latent_dim = latent_dim
        self.volumeSize = None


        self.encoder_input, self.encoder_output, self.encoder = self.create_encoder()
        self.decoder_input, self.decoder_output, self.decoder = self.create_decoder()
        self.autoencoder_input, self.autoencoder_output, self.autoencoder = self.create_autoencoder()


        self.__call__ = self.autoencoder.predict
        self.fit = self.autoencoder.fit
        self.fit_generator = self.autoencoder.fit_generator

        # Compiling models

        # self.encoder.compile(
        #     optimizer=keras.optimizers.Adagrad(),
        #     loss=keras.losses.SparseCategoricalCrossentropy(),
        #     metrics=[keras.metrics.Accuracy()],
        # )
        # self.decoder.compile(
        #     optimizer=keras.optimizers.Adagrad(),
        #     loss=keras.losses.SparseCategoricalCrossentropy(),
        #     metrics=[keras.metrics.Accuracy()],
        # )

        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss='mse',
            metrics=[keras.metrics.Accuracy()],
            run_eagerly=False
        )


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()





def training():
    image_size = (28, 28, 1)
    batch_size = 256

    train = keras.preprocessing.image_dataset_from_directory("PetImages",
                                                                        image_size=image_size[:2],
                                                                        seed=123,
                                                                        validation_split=0.2,
                                                                        subset="training",
                                                                        batch_size=batch_size
                                                             )
    (train_x,train_y),(val_x, val_y) = mnist.load_data()
    print(train_x.shape)
    train_x = train_x.reshape((-1, 28, 28, 1))
    print(train_x.shape)
    #train = train.map((lambda x, y: (x, x)))
    train = tf.data.Dataset.from_tensor_slices((train_x, train_x))
    train = train.batch(batch_size=batch_size)

    autoenc = AutoEncoder(input_shape=image_size)
    autoenc.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint("model_checkpoint/autoenv"),
        keras.callbacks.TensorBoard("logs", write_images=True)
    ]

    autoenc.autoencoder.fit(train_x, train_x, epochs=50, shuffle=True, callbacks=callbacks)

    autoenc.autoencoder.save("models/autoenc")
    autoenc.encoder.save("models/encoder")
    autoenc.decoder.save("models/decoder")

    # plot_images(train_ds[0])
    # plt.show()


def validation():
    image_size = (28, 28, 1)
    batch_size = 64

    train = keras.preprocessing.image_dataset_from_directory("PetImages",
                                                             image_size=image_size[:2],
                                                             seed=123,
                                                             validation_split=0.2,
                                                             subset="validation",
                                                             #batch_size=batch_size
                                                             )
    autoenc = keras.models.load_model("models/mnist/autoenc")
    encoder = keras.models.load_model("models/mnist/encoder")
    decoder = keras.models.load_model("models/mnist/decoder")

    image = tf.convert_to_tensor([keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img("PetImages/Cat/0.jpg",target_size=image_size))])

    (train_x, train_y), (val_x, val_y) = mnist.load_data()

    # train = train.map((lambda x, y: (x, x)))
    # train = tf.data.Dataset.from_tensor_slices((train_x, train_x))

    autoenc.evaluate(train_x, train_x)
    image = train_x
    prediction = autoenc.predict(image)
    compressed = encoder.predict(image)
    uncompressed = decoder.predict(compressed)

    plt.gray()
    plot_images(image/255)
    plt.show()


    plt.gray()
    plot_images(prediction/255)
    plt.show()

    plot_images(uncompressed.reshape(-1, 4, 4))
    plt.show()

    for _ in range(100):
        randomcompressed = np.random.random((16, 16))
        print(np.max(randomcompressed))
        print(np.max(compressed))
        randoms = decoder(randomcompressed*165)
        plot_images(randoms)
        plt.show()

if __name__ == '__main__':
    # training()
    validation()