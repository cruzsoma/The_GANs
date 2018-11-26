import tensorflow as tf
import numpy as np
import glob
import os
from imageio import imread, imsave, mimsave
from tools.gif import generate_gif

from scipy.misc import imresize

learning_rate = 0.0002
beta1 = 0.5
batch_size = 100
z_dim = 100
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

total_steps = 500

MODEL_NAME = 'DCGAN'
MODELS_DIR = 'models/DCGAN'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

IMAGE_GEN_FOLDER = 'generated_images'
if not os.path.exists(os.path.join(MODELS_DIR, IMAGE_GEN_FOLDER)):
    os.mkdir(os.path.join(MODELS_DIR, IMAGE_GEN_FOLDER))

dataset_path = 'C:/MachineLearning/TrainingData/celeba'
images = glob.glob(os.path.join(dataset_path, '*.*'))

tf.logging.set_verbosity(tf.logging.INFO)

def read_image_by_path(path, img_height, img_width):
    image = imread(path)
    height = image.shape[0]
    width = image.shape[1]
    if height > width:
        image = image[height // 2 - width // 2: height // 2 + width // 2, :, :]
    else:
        image = image[:, width // 2 - height // 2: width // 2 + height // 2, :]
    image = imresize(image, (img_height, img_width))
    return image / 255.

def combine_images(images):
    if isinstance(images, list):
        images = np.array(images)

    height = images.shape[1]
    width = images.shape[2]
    n = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n + n + 1,
             images.shape[2] * n + n + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n + n + 1,
             images.shape[2] * n + n + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n + n + 1,
             images.shape[2] * n + n + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(images.shape))
    for i in range(n):
        for j in range(n):
            # n[i * n + j]  means n[i][j]
            filter = i * n + j
            if filter < images.shape[0]:
                m[1 + i + i * height:1 + i + (i + 1) * height, 1 + j + j * width:1 + j + (j + 1) * width] = images[filter]
    return m

def leaky_relu(x, leak = 0.2):
    return tf.maximum(x, leak * x);

def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

# neuron network
X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def discriminator(image, reuse=None, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        layer0 = leaky_relu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))
        layer1 = tf.layers.conv2d(layer0, kernel_size=5, filters=128, strides=2, padding='same')
        layer1 = leaky_relu(tf.layers.batch_normalization(layer1, training=is_training, momentum=momentum))
        layer2 = tf.layers.conv2d(layer1, kernel_size=5, filters=256, strides=2, padding='same')
        layer2 = leaky_relu(tf.layers.batch_normalization(layer2, training=is_training, momentum=momentum))
        layer3 = tf.layers.conv2d(layer2, kernel_size=5, filters=512, strides=2, padding='same')
        layer3 = leaky_relu(tf.layers.batch_normalization(layer3, training=is_training, momentum=momentum))
        layer4 = tf.layers.flatten(layer3)
        layer4 = tf.layers.dense(layer4, units=1)
        return tf.nn.sigmoid(layer4), layer4

def generator(input, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 4
        layer0 = tf.layers.dense(input, units=d * d * 512)
        layer0 = tf.reshape(layer0, shape=[-1, d, d, 512])
        layer0 = tf.nn.relu(tf.layers.batch_normalization(layer0, training=is_training, momentum=momentum))
        layer1 = tf.layers.conv2d_transpose(layer0, kernel_size=5, filters=256, strides=2, padding='same')
        layer1 = tf.nn.relu(tf.layers.batch_normalization(layer1, training=is_training, momentum=momentum))
        layer2 = tf.layers.conv2d_transpose(layer1, kernel_size=5, filters=128, strides=2, padding='same')
        layer2 = tf.nn.relu(tf.layers.batch_normalization(layer2, training=is_training, momentum=momentum))
        layer3 = tf.layers.conv2d_transpose(layer2, kernel_size=5, filters=64, strides=2, padding='same')
        layer3 = tf.nn.relu(tf.layers.batch_normalization(layer3, training=is_training, momentum=momentum))
        layer4 = tf.layers.conv2d_transpose(layer3, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh, name='generated')
        return layer4

gen = generator(noise)
dis_real, dis_real_logits = discriminator(X)
dis_fake, dis_fake_logits = discriminator(gen, reuse=True)

variables_gen = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
variables_dis = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

loss_dis_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(dis_real_logits, tf.ones_like(dis_real)))
loss_dis_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(dis_fake_logits, tf.zeros_like(dis_fake)))
loss_dis = loss_dis_real + loss_dis_fake

loss_gen = tf.reduce_mean(sigmoid_cross_entropy_with_logits(dis_fake_logits, tf.ones_like(dis_fake)))

global_step = tf.Variable(1, name='global_step', trainable=False)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss_dis, global_step=global_step, var_list=variables_dis)
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss_gen, var_list=variables_gen)

saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

last_file = tf.train.latest_checkpoint(MODELS_DIR)
if last_file:
    tf.logging.info('Restoring model from {}'.format(last_file))
    saver.restore(sess, last_file)

noise_image = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
samples = []
loss = {'dis': [], 'gen': []}

offset = 0
step = sess.run(global_step)
for i in range(step, 60000):
    train_noise = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    offset = (batch_size * i) % len(images)
    batch = np.array([read_image_by_path(img, IMAGE_HEIGHT, IMAGE_WIDTH) for img in images[offset: offset + batch_size]])
    batch = (batch - 0.5) * 2

    dis_ls, gen_ls = sess.run([loss_dis, loss_gen], feed_dict={X:batch, noise: train_noise, is_training: True})
    loss['dis'].append(dis_ls)
    loss['gen'].append(gen_ls)

    sess.run(optimizer_dis, feed_dict={X: batch, noise: train_noise, is_training: True})
    sess.run(optimizer_gen, feed_dict={X: batch, noise: train_noise, is_training: True})
    sess.run(optimizer_gen, feed_dict={X: batch, noise: train_noise, is_training: True})

    if i % 10 == 0:
        tf.logging.info('step: %d,  Discriminator Loss %f, Generator Loss %f' % (i, dis_ls, gen_ls))
    if i % 50 == 0:
        generated_images = sess.run(gen, feed_dict={noise: noise_image, is_training: False})
        generated_images = (generated_images + 1) / 2
        generated_images = combine_images([img[:, : , :] for img in generated_images])
        imsave(os.path.join(MODELS_DIR, IMAGE_GEN_FOLDER, 'sample_%d.jpg' % i), generated_images)
        samples.append(generated_images)
        saver.save(sess, os.path.join(MODELS_DIR, 'dcgan.ckpt'), global_step=i)

tf.logging.info('Done DCGAN training')
saver.save(sess, os.path.join(MODELS_DIR, 'dcgan.ckpt'), global_step=total_steps)
generate_gif(MODEL_NAME, skip_count=20, fps=10)