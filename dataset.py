import cv2
import albumentations as A
import numpy as np
import os
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

####################
### make dataset ###
####################
class Dataset():
    def __init__(self, opt, use_list):
        self.img_size = opt.img_size
        self.channels = opt.channels
        self.class_list = use_list
        self.data_aug = opt.aug

        if self.data_aug:
            elastictransform = A.ElasticTransform(alpha=1, 
                                                sigma=10, 
                                                alpha_affine=20, 
                                                interpolation=1, 
                                                border_mode=0, 
                                                value=None, 
                                                mask_value=None, 
                                                always_apply=False, 
                                                approximate=False, 
                                                same_dxdy=True, 
                                                p=0.5)

            resize = A.Resize(height=opt.img_size, width=opt.img_size, p=1)
            aug_list = [elastictransform, resize]

            self.transform = A.Compose(aug_list)

    def get_bounding_box(self, contours, img_size):
        #戻り値の座標  [x, y, w, h]
        coordinate = [img_size, img_size, 0, 0]

        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])

            if x<25 or x+w==img_size or y<25 or y+h==img_size:
                pass
            else:
                if x  < coordinate[0]:
                    coordinate[0] = x

                if y < coordinate[1]:
                    coordinate[1] = y

                if x + w > coordinate[2]:
                    coordinate[2] = x + w 
                if y + h > coordinate[3]:
                    coordinate[3] = y + h 

        return coordinate

    def adjust_square(self, img, brank=0):
        h, w = img.shape
        if h>=w:
            img = cv2.copyMakeBorder(img, 0, 0, int((h-w)/2), int((h-w)/2), cv2.BORDER_CONSTANT)
        else:
            img = cv2.copyMakeBorder(img, int((w-h)/2), int((w-h)/2), 0, 0, cv2.BORDER_CONSTANT)
        # 正方形にする
        img = cv2.copyMakeBorder(img, brank, brank, brank, brank, cv2.BORDER_CONSTANT)

        return img

    def load_data(self, files):
        xs = [] # 画像
        ys = [] # ラベル

        for f in files:
            f = bytes.decode(f.numpy())
            label = self.class_list.index(f.split('/')[2])

            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(self.img_size, self.img_size))
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 輪郭抽出
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 文字領域の座標取得
            coordinate = self.get_bounding_box(contours, self.img_size)
            # 入力画像のクロップ
            img = img[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
            # 正方形に調整
            img = self.adjust_square(img, 10)

            # # データ拡張
            if self.aug:
                img = cv2.bitwise_not(img)
                transformed = self.transform(image=img)
                img = transformed["image"]
                img = cv2.bitwise_not(img)
            img = cv2.bitwise_not(img) # 白黒反転

            if self.channels==3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = img /255

            xs.append(img)
            ys.append(label)
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.int32)

        return xs, ys

    def load_test_data(self,files):
        xs = [] # 画像
        ys = [] # ラベル

        for f in files:
            f = bytes.decode(f.numpy())
            label = self.class_list.index(f.split('/')[2])
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(self.img_size, self.img_size))
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 輪郭抽出
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 文字領域の座標取得
            coordinate = self.get_bounding_box(contours, self.img_size)
            # 入力画像のクロップ
            img = img[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
            # 正方形に調整
            img = self.adjust_square(img, 10)
            img = cv2.resize(img,(self.img_size, self.img_size))
            img = cv2.bitwise_not(img) # 白黒反転

            if self.channels==3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = img /255

            xs.append(img)
            ys.append(label)
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.int32)

        return xs, ys