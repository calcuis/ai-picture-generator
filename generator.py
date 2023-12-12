import tensorflow as tf
import matplotlib.pyplot as plt
import random

seed = random.getrandbits(32)
n_images = 25
codings_size = 100
generator = tf.keras.models.load_model("./models/")

def generate(generator, seed):
    noise = tf.random.normal(shape=[n_images, codings_size], seed=seed)
    generated_images = generator(noise, training=False)

    fig = plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        # plt.subplot(6, 6, i+1)
        plt.subplot(5, 5, i+1)
        plt.imshow(generated_images[i, :, :, :])
        plt.axis('off')
    plt.savefig("5x5_samples.png")
    
generate(generator, seed)
