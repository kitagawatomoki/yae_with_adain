import glob2
import os
import numpy as np
import tensorflow as tf
import cv2

def get_data(data_root, class_list):
    train_list = []
    predict_list = []
    for char in class_list:
        train_path = glob2.glob(os.path.join(data_root, char, "**", "*.png")) + glob2.glob(os.path.join(data_root, char, "**", "*.jpg"))
        train_list +=train_path
        predict_list.append(train_path[5])

    return train_list, predict_list

def calculate_ssim_loss(img, decoded):
    return 1-tf.reduce_mean(tf.image.ssim(img, decoded, 
                                          max_val = 1.0,filter_size=11,
                                        filter_sigma=1.5, k1=0.01, k2=0.03 ))
    # loss = tf.reduce_mean(tf.abs(tf.subtract(decoded, img) + 1.0e-15))
    # # loss = tf.reduce_mean(tf.keras.losses.MSE(img, decoded))
    # # loss = tf.reduce_mean(tf.square(tf.subtract(img, decoded)))
    # # loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(img, decoded)))
    # return loss

def calculate_mse_loss(img, decoded):
    # loss = tf.reduce_mean(tf.abs(tf.subtract(decoded, img) + 1.0e-15))
    loss = tf.reduce_mean(tf.keras.losses.MSE(img, decoded))
    # # loss = tf.reduce_mean(tf.square(tf.subtract(img, decoded)))
    # # loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(img, decoded)))
    return loss

def calculate_cross_entropy(label, pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(label, pred)
    return loss

def calculate_implicit(r_style, l_style):
    # loss = tf.reduce_mean(tf.norm(r_style - l_style + 1.0e-15, ord='euclidean', axis=1))
    loss = tf.reduce_mean(tf.norm(tf.subtract(r_style, l_style) + 1.0e-15, ord=2, axis=1))
    return loss



def generate_images(predict_data, model, epoch, file_writer, num_gen, class_list, opt):
    num_character = len(class_list)
    if num_gen>num_character:
        num_gen = num_character

    for img ,label in predict_data:
        style, _ = model.encoder.predict(img)
        style = np.array(style, dtype=np.float32)
        styles = np.tile(style, (num_gen, 1))
        s_split = np.array_split(styles, int(num_gen/opt.batch_size)+1, 0)

        sample_label = np.arange(num_gen)
        l_split = np.array_split(sample_label, int(num_gen/opt.batch_size)+1)
        l = int(label.numpy())

        outputs = []
        for split in range(int(num_gen/opt.batch_size)+1):
            if split ==0:
                out_img = model.decoder([s_split[split], l_split[split]],training=False)
                outputs = np.array(out_img, dtype=np.float32)
            else:
                out_img = model.decoder([s_split[split], l_split[split]],training=False)
                outputs = np.concatenate([outputs, np.array(out_img, dtype=np.float32)], 0)
        outputs = np.array(outputs, dtype=np.float32)

        i = np.array(img, dtype=np.float32)
        i = i.reshape([opt.img_size, opt.img_size, opt.channels])
        i = np.concatenate([i, np.ones((opt.img_size, opt.img_size*9, opt.channels)) ], 1)

        if num_gen%10 == 0:
            column = int(num_gen/10) 
            row = 10
            add_canvas = 0
        else:
            column = 1 + int(num_gen/10) 
            row = 10
            add_canvas = 10 - num_gen%10

        canvas = np.ones((add_canvas,opt.img_size, opt.img_size, opt.channels)) 
        outputs = np.concatenate([outputs, canvas], 0)

        if column==1:
            imgs =outputs[0]
            for n in range(9):
                imgs = np.concatenate([imgs, outputs[n+1]], 1)

        else:
            for n in range(column):
                img_column = outputs[10*n]
                for m in range(9):
                    img_column = np.concatenate([img_column, outputs[10*n+m+1]], 1)

                if n ==0:
                    imgs = img_column
                else:
                    imgs = np.concatenate([imgs, img_column], 0)

        imgs = np.concatenate([i, imgs], 0)

        if epoch % 10 == 0:
            save_path = os.path.join(opt.result, "images", "ep_{}_{}.png".format(epoch, class_list[l]))
            cv2.imwrite(save_path, imgs*255)

            with file_writer.as_default():
                tf.summary.image("%s"%(class_list[l]), tf.expand_dims(imgs, 0), step=epoch)