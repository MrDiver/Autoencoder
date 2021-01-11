# Keras Notes

- Links
- https://keras.io/getting_started/intro_to_keras_for_engineers/
- https://keras.io/guides/functional_api/
- https://keras.io/guides/training_with_built_in_methods/
- https://keras.io/api/callbacks/
- https://keras.io/examples/vision/image_classification_from_scratch/
## Steps

### Imports 
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
```

### Loading a Dataset
- Uses Python to load data from Directories to *Tensorflow Datasets*
    ```python
    tf.keras.preprocessing.image_dataset_from_directory
    ```
  
- In addition, the TensorFlow tf.data includes other similar utilities, such as tf.data.experimental.make_csv_dataset to load structured data from CSV files.
    
- Example Loading
    ```python
    # Create a dataset.
    dataset = keras.preprocessing.image_dataset_from_directory(
      'path/to/main_directory', batch_size=64, image_size=(200, 200))
    
    # For demonstration, iterate over the batches yielded by the dataset.
    for data, labels in dataset:
       print(data.shape)  # (64, 200, 200, 3)
       print(data.dtype)  # float32
       print(labels.shape)  # (64,)
       print(labels.dtype)  # int32
    ```
- This can also be accomplished with the `ÌmageDataGenerator class`

```python
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)
```

### Preprocessing
- How the data should be normalized
    - Tokenization of string data, followed by token indexing.
    - Feature normalization.
    - Rescaling the data to small values (in general, input values to a neural network should be close to zero -- typically we expect either data with zero-mean and unit-variance, or data in the [0, 1] range.

- Keras has Preprocessing Layers to build an End to End model
    - Vectorizing raw strings of text via the `TextVectorization` layer
    - Feature normalization via the `Normalization` layer
    - Image rescaling, cropping, or image data augmentation

- Some preprocessing layers have a state:

    - `TextVectorization` holds an index mapping words or tokens to integer indices
    - `Normalization` holds the mean and variance of your features

    - **The state of a preprocessing layer is obtained by calling `layer.adapt(data)` on a sample of the training data (or all of it).**
    
#### Example on Text Vectorization
```python
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Example training data, of dtype `string`.
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# Create a TextVectorization layer instance. It can be configured to either
# return integer token indices, or a dense token representation (e.g. multi-hot
# or TF-IDF). The text standardization and text splitting algorithms are fully
# configurable.
vectorizer = TextVectorization(output_mode="int")

# Calling `adapt` on an array or dataset makes the layer generate a vocabulary
# index for the data, which can then be reused when seeing new data.
vectorizer.adapt(training_data)

# After calling adapt, the layer is able to encode any n-gram it has seen before
# in the `adapt()` data. Unknown n-grams are encoded via an "out-of-vocabulary"
# token.
integer_data = vectorizer(training_data)
print(integer_data)
```

#### Example on Data Normalization
```python
from tensorflow.keras.layers.experimental.preprocessing import Normalization

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))
```

##### Example on Image Preprocessing
- The `Rescaling` and `CenterCrop` Layers are stateless so theres no need to call the `adapt()` function

```python
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)

output_data = scaler(cropper(training_data))
print("shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))
```


### Building Models with the *Functional API*

A layer is just an Input->Output Tranformation of Data
```python
dense = keras.layers.Dense(units=16)
```
You can specify `shape` and `dtype` for every layer

Therefore you can define your input as follows
```python
# Let's say we expect our inputs to be RGB images of arbitrary size
inputs = keras.Input(shape=(None, None, 3))
```
After that you can specify how the model should look with Preprocessing

```python
from tensorflow.keras import layers

# Center-crop images to 150x150
x = CenterCrop(height=150, width=150)(inputs)
# Rescale images to [0, 1]
x = Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)

# Add a dense classifier on top
num_classes = 10
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

Now you can create a Model like this

```python
model = keras.Model(inputs=inputs, outputs=outputs)
```

### Training the model

After creating the model we can compile it with a loss function and optimizers

```python
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.CategoricalCrossentropy())
```


A Full example of loading Data creating a Model and Fitting it

```python
# Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Build a simple model
inputs = keras.Input(shape=(28, 28))
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model for 1 epoch from Numpy data
batch_size = 64
print("Fit on NumPy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

# Train the model for 1 epoch using a dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
print("Fit on Dataset")
history = model.fit(dataset, epochs=1)
```

### Monitoring the Training of the Model with Keras Metrics

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
history = model.fit(dataset, epochs=1)
```

```python
938/938 [==============================] - 1s 929us/step - loss: 0.0835 - acc: 0.9748
```

We can also pass validation data to fit directly

```python
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(dataset, epochs=1, validation_data=val_dataset)
```

```python
938/938 [==============================] - 1s 1ms/step - loss: 0.0563 - acc: 0.9829 - val_loss: 0.1041 - val_acc: 0.9692
```

### Keras Callbacks for Checpointing and Training Scripting

We can use callbacks to give actions in the Training process.
For example you can save the model after every epoch to make sure that no progress is lost if the training fails.

```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='path/to/my/model_{epoch}',
        save_freq='epoch')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
```

You can also use TensorBoard during Training

```python
callbacks = [
    keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
```

### Model Evaluation

After the model is trained on the dataset you can quickly analyze the performance with the evaluation.

```python
loss, acc = model.evaluate(val_dataset)  # returns loss and metrics
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)
```

### Custom Training Step

```python
class CustomModel(keras.Model):
  def train_step(self, data):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)  # Forward pass
      # Compute the loss value
      # (the loss function is configured in `compile()`)
      loss = self.compiled_loss(y, y_pred,
                                regularization_losses=self.losses)
    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=[...])

# Just use `fit` as usual
model.fit(dataset, epochs=3, callbacks=...)
```