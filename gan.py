import os
# Reduce TensorFlow log noise (optional: 0=all, 1=no INFO, 2=no WARNING, 3=errors only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Report whether GPU (CUDA) is available
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"Using GPU: {[gpu.name for gpu in gpus]}")
else:
    print("Using CPU (no GPU detected)")

# Load and preprocess CIFAR-10 data (32x32 RGB)
(train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
BUFFER_SIZE = 50000
BATCH_SIZE = 256
NOISE_DIM = 100
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

# Output directory for generated images
CHECKPOINT_DIR = 'generated_images'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Generator model (32x32x3) with additional convolutional layers
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((4, 4, 512)),
        layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'),
    ])
    return model


# Discriminator model (32x32x3)
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1),
    ])
    return model


# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def save_generated_images(generator, epoch, num_images=16, seed=42):
    """Generate and save a grid of images to disk."""
    tf.random.set_seed(seed)
    noise = tf.random.normal([num_images, NOISE_DIM])
    generated = generator(noise, training=False)
    generated = (generated + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
    generated = tf.clip_by_value(generated, 0.0, 1.0)
    n = int(num_images ** 0.5)
    fig, axes = plt.subplots(n, n, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        if generated.shape[-1] == 1:
            ax.imshow(generated[i].numpy().squeeze(), cmap='gray')
        else:
            ax.imshow(generated[i].numpy())
        ax.axis('off')
    plt.savefig(os.path.join(CHECKPOINT_DIR, f'epoch_{epoch:04d}.png'), bbox_inches='tight')
    plt.close()
    print(f'Saved generated images to {CHECKPOINT_DIR}/epoch_{epoch:04d}.png')


# Optimizers
generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Training function
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss


# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        num_batches = 0
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            epoch_gen_loss += float(gen_loss)
            epoch_disc_loss += float(disc_loss)
            num_batches += 1
        avg_gen = epoch_gen_loss / num_batches
        avg_disc = epoch_disc_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs} — gen_loss: {avg_gen:.4f}, disc_loss: {avg_disc:.4f}")
        if (epoch + 1) % 10 == 0:
            save_generated_images(generator, epoch + 1)


# Run the training
train(train_dataset, epochs=50)
