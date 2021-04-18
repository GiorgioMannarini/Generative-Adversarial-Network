import tensorflow as tf
import matplotlib.pyplot as plt
from classes.model import GenerativeAdversarialNetwork

BATCH_SIZE = 256
EPOCHS = 50
NOISE_DIM = 100
NUMBER_OF_PREDS = 16


# We don't have a test. Unsupervised learning
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshaping from 60k x 28 x 28 into 60k x 28 x 28 x 1 because conv2D needs for each image three dims
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

# Normalizing between -1 and 1 (that's why we have tanh at the end of the generator)
train_images = (train_images - 127.5)/127.5

# tf dataset from the array
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(BATCH_SIZE)

# Defining the model
GAN = GenerativeAdversarialNetwork(image_size=[28, 28, 1], latent_dim=NOISE_DIM, batch_size=BATCH_SIZE,
                                   checkpoint_dir='./checkpoints')

# Sample of a mnist image
plt.imshow(train_images[0, :, :, 0], cmap='gray')
plt.show()
plt.close()

GAN.train(train_dataset, EPOCHS, from_pretrained=True)
