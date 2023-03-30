import argparse
import shutil
import random
import time
import tensorflow as tf

from model import YAE
from dataset import Dataset
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-data_root', type=str, default="dataset")
parser.add_argument('-use_class', type=str, default="class.txt")
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-num_epoch', type=int, default=1000)
parser.add_argument('-img_size', type=int, default=256)
parser.add_argument('-channels', type=int, default=1)
parser.add_argument('-aug', action='store_true')
parser.add_argument('-num_predict', type=int, default=500)
args = parser.parse_args()

if os.path.exists(os.path.join(args.result, args.log)):
    shutil.rmtree(os.path.join(args.result, args.log))
os.makedirs(os.path.join(args.result, "log"), exist_ok=True)
os.makedirs(os.path.join(args.result, "images"), exist_ok=True)
os.makedirs(os.path.join(args.result, "model"), exist_ok=True)
os.makedirs(os.path.join(args.result, "chp"), exist_ok=True)

GPU_ID = args.gpu
# ========== GPU周りの設定 =========== #
gpus = tf.config.experimental.list_physical_devices("GPU")
[tf.config.experimental.set_memory_growth(
    gpu, enable=True) for gpu in gpus]

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[GPU_ID], "GPU")
tf.config.experimental.set_memory_growth(physical_devices[GPU_ID], True)


################
### Use data ###
################
with open(args.use_class) as f:
    class_list = [s.strip() for s in f.readlines()]

print('num class : ', len(class_list))
print('use kanji : ')
print(class_list)

train_list, predict_list = get_data(args.data_root, args.use_class)

print('num train data : ',len(train_list))
predict_list = random.sample(predict_list, 5)

####################
### Make dataset ###
####################sou
d = Dataset(args, class_list)
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset_x = tf.data.Dataset.from_tensor_slices(train_list)
train_dataset_x = train_dataset_x.shuffle(len(train_list)).batch(args.batch_size, drop_remainder=True)
train_dataset_xy = train_dataset_x.map(lambda x: tf.py_function(d.load_data, [x], Tout=[tf.float32, tf.int32, tf.int32]))
train_dataset_xy = train_dataset_xy.prefetch(buffer_size=AUTOTUNE)

predict_data_x = tf.data.Dataset.from_tensor_slices(predict_list)
predict_data_x = predict_data_x.batch(1)
predict_data_xy = predict_data_x.map(lambda x: tf.py_function(d.load_test_data, [x], Tout=[tf.float32, tf.int32]))
predict_data_xy = predict_data_xy.prefetch(buffer_size=AUTOTUNE)

####################
### Define model ###
####################
model = YAE(args, len(class_list))
# model.encoder.summary()
# model.decoder.summary()

for img, label, label_id in train_dataset_xy.take(1):
    style, content= model.encoder(img)
    print(style.shape)
    print(content.shape)
    output = model.decoder([style, label])
    print(output.shape)

train_summary_writer = tf.summary.create_file_writer(os.path.join(args.result, args.log))
# optimizer = tf.keras.optimizers.Adam(args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
optimizer = tf.keras.optimizers.Adam(args.lr)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,net=model)
manager = tf.train.CheckpointManager(checkpoint, os.path.join(args.result, args.checkpoint), max_to_keep=3)

lambda_e = 1
lambda_i = 1
lambda_c = 1

generate_images(predict_data_xy ,
                model,
                0,
                train_summary_writer,
                500,
                class_list,
                args)

for epoch in range(args.num_epoch+1):
    start = time.time()

    loss_av = tf.keras.metrics.Mean()
    loss_classification_av = tf.keras.metrics.Mean()
    loss_exlicit_av = tf.keras.metrics.Mean()
    loss_implicit_av = tf.keras.metrics.Mean()
    loss_mse_av = tf.keras.metrics.Mean()
    loss_ssim_av = tf.keras.metrics.Mean()

    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for img, label, label_id in train_dataset_xy:

        with tf.GradientTape(persistent=True) as tape:

            style, content = model.encoder(img, training=True)
            left_output = model.decoder([style, label], training=True)
            l_style, l_content = model.encoder(left_output, training=True)

            ##Reconstruction loss
            loss_mse = calculate_mse_loss(img, left_output)
            loss_ssim= calculate_ssim_loss(img, left_output)

            ##Classification loss
            loss_classification = calculate_cross_entropy(label, content)

            ##Accuracy
            label = tf.reshape(label, [args.batch_size, 1])
            accuracy_classification = accuracy(label, content)

            random_label0 = tf.random.uniform(shape=[args.batch_size]
                                           , minval=0, maxval=len(class_list)
                                           , dtype=tf.int32)

            right_output0 = model.decoder([style, random_label0], training=True)
            # right_output1 = model.decoder([style, random_label1], training=True)
            r_style0, r_content0 = model.encoder(right_output0, training=True)
            # r_style1, r_content1 = model.encoder(right_output1, training=True)

            ##Explicit loss(明示的)
            loss_explicit0 = calculate_cross_entropy(random_label0, r_content0)
            # loss_explicit1 = calculate_cross_entropy(random_label1, r_content1)
            loss_explicit = loss_explicit0 #+ loss_explicit1

            ##Implicit loss(暗黙的)
            loss_implicit0 = calculate_implicit(r_style0, l_style)
            # loss_implicit1 = calculate_implicit(r_style1, l_style)
            loss_implicit = loss_implicit0 #+ loss_implicit1

            ##Global Loss
            loss = loss_mse + 0.5*loss_ssim  + loss_classification + lambda_e*loss_explicit + lambda_i*loss_implicit

        # Calculate the gradients for encoder and decoder
        gradients = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the optimizer
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_av(loss)
        loss_classification_av(loss_classification)
        loss_exlicit_av(loss_explicit)
        loss_implicit_av(loss_implicit)
        loss_mse_av(loss_mse)
        loss_ssim_av(loss_ssim)
        accuracy(label, content)




        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_av.result(), step=epoch)
            tf.summary.scalar('loss_classification', loss_classification_av.result(), step=epoch)
            tf.summary.scalar('loss_explicit', loss_exlicit_av.result(), step=epoch)
            tf.summary.scalar('accuracy_classification', accuracy.result(), step=epoch)
            tf.summary.scalar('loss_implicit', loss_implicit_av.result(), step=epoch)
            tf.summary.scalar('loss_mse', loss_mse_av.result(), step=epoch)
            tf.summary.scalar('loss_ssim', loss_ssim_av.result(), step=epoch)

    if epoch % 10 == 0:
        model.encoder.save('%s/encoder_%d.h5' %(os.path.join(args.result, args.output_model),epoch), True)
        model.decoder.save('%s/decoder_%d.h5' %(os.path.join(args.result, args.output_model), epoch), True)

    generate_images(predict_data_xy ,
                    model,
                    epoch,
                    train_summary_writer,
                    args.num_predict,
                    class_list,
                    args)
    manager.save()
    print('Time taken for epoch {} is {} sec'.format(epoch,time.time()-start))
    print('loss: {:.5f}'.format(loss_av.result()))
    print('accuracy : {:.5f} loss_classification: {:.5f} '.format(accuracy.result(),loss_classification_av.result()))
    print('loss_explict : {:.5f} '.format(loss_exlicit_av.result()))
    print('loss_implict : {:.5f} '.format(loss_implicit_av.result()))
    print('loss_mse : {:.5f}'.format(loss_mse_av.result()))
    print('loss_ssim : {:.5f}'.format(loss_ssim_av.result()))