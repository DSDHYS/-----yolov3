from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from yolo_training import Generator
import cv2
from PIL import Image, ImageDraw

class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size, is_train):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a


    def merge_bboxes(self,bboxes, cutx, cuty):

        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                        if y2-y1 < 5:
                            continue
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                        if x2-x1 < 5:
                            continue
                    
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                        if y2-y1 < 5:
                            continue
                    
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                        if x2-x1 < 5:
                            continue

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                        if y2-y1 < 5:
                            continue

                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                        if x2-x1 < 5:
                            continue

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                        if y2-y1 < 5:
                            continue

                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                        if x2-x1 < 5:
                            continue

                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox
    def get_random_data(self,annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True):
        '''random preprocessing for real-time data augmentation'''
        h, w = input_shape
        min_offset_x = 0.4
        min_offset_y = 0.4
        scale_low = 1-min(min_offset_x,min_offset_y)
        scale_high = scale_low+0.2

        image_datas = [] 
        box_datas = []
        index = 0
        place_x = [0,0,int(w*min_offset_x),int(w*min_offset_x)]
        place_y = [0,int(h*min_offset_y),int(w*min_offset_y),0]
        #print(annotation_line)
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            #print(line)
            # 打开图片
            image = Image.open(line_content[0])
            image = image.convert("RGB") 
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            #print(box)
            
            # image.save(str(index)+".jpg")
            # 是否翻转图片
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            # 对输入进来的图片进行缩放
            new_ar = w/h
            scale = self.rand(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw,nh), Image.BICUBIC)

            # 进行色域变换
            hue =self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
            val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
            x = rgb_to_hsv(np.array(image)/255.)
            x[..., 0] += hue
            x[..., 0][x[..., 0]>1] -= 1
            x[..., 0][x[..., 0]<0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x>1] = 1
            x[x<0] = 0
            image = hsv_to_rgb(x)

            image = Image.fromarray((image*255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            #print(index)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255

            # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
            # print('============')
            index = index + 1
            # print(index)
            # print('============')            
            box_data = []
            # 对box进行重新处理
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

            img = Image.fromarray((image_data*255).astype(np.uint8))
            for j in range(len(box_data)):
                thickness = 3
                left, top, right, bottom  = box_data[j][0:4]
                draw = ImageDraw.Draw(img)
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(255,255,255))
            #img.show()

        
        # 将图片分割，放在一起
        cutx = np.random.randint(int(w*min_offset_x), int(w*(1 - min_offset_x)))
        cuty = np.random.randint(int(h*min_offset_y), int(h*(1 - min_offset_y)))

        # print('||',cutx,'||')
        # print('|',cuty,'|')
        new_image = np.zeros([h,w,3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]


        # 对框进行进一步的处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        #return new_image, new_boxes
        return image_data, box_data,new_image,new_boxes


    def get_random_data_val(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
            """实时数据增强的随机预处理"""
            line = annotation_line.split()
            #print(line)
            image = Image.open(line[0])
            iw, ih = image.size
            h, w = input_shape
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

            if not random:
                scale = min(w/iw, h/ih)
                nw = int(iw*scale)
                nh = int(ih*scale)
                dx = (w-nw)//2
                dy = (h-nh)//2

                image = image.resize((nw,nh), Image.BICUBIC)
                new_image = Image.new('RGB', (w,h), (128,128,128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image, np.float32)

                # 调整目标框坐标
                box_data = np.zeros((len(box), 5))
                if len(box) > 0:
                    np.random.shuffle(box)
                    box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                    box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                    box[:, 0:2][box[:, 0:2] < 0] = 0
                    box[:, 2][box[:, 2] > w] = w
                    box[:, 3][box[:, 3] > h] = h
                    box_w = box[:, 2] - box[:, 0]
                    box_h = box[:, 3] - box[:, 1]
                    box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
                    box_data = np.zeros((len(box), 5))
                    box_data[:len(box)] = box

                return image_data, box_data
                
            # 调整图片大小
            new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.25, 2)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 放置图片
            dx = int(self.rand(0, w - nw))
            dy = int(self.rand(0, h - nh))
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image = new_image

            # 是否翻转图片
            flip = self.rand() < .5
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # 色域变换
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue*360
            x[..., 0][x[..., 0]>1] -= 1
            x[..., 0][x[..., 0]<0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:,:, 0]>360, 0] = 360
            x[:, :, 1:][x[:, :, 1:]>1] = 1
            x[x<0] = 0
            image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

            # 调整目标框坐标
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                if flip:
                    box[:, [0, 2]] = w - box[:, [2, 0]]
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box
                
            return image_data, box_data
    def __getitem__(self, index):
        lines = self.train_lines
        n = self.train_batches
        # print(n)
        # print(index)
        if index+4>n and index<n:
            index=index-4
        index = index % n
        #print(index)

        #print(lines[index],'DSD')
        # n = self.train_batches
        # index = index % n
        # if self.is_train:
        #     img, y = self.get_random_data(lines[index], self.image_size[0:2])
        # else:
        #     img, y = self.get_random_data(lines[index], self.image_size[0:2], False)
        if self.is_train:
            img, y,image_data, box_data = self.get_random_data(lines[index:index+4], self.image_size[0:2])
            img = Image.fromarray((image_data*255).astype(np.uint8))
            for j in range(len(box_data)):
                thickness = 3
                left, top, right, bottom  = box_data[j][0:4]
                draw = ImageDraw.Draw(img)
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(255,255,255))
            #img.show()
            if len(y) != 0:
                # 从坐标转换成0~1的百分比
                boxes = np.array(y[:, :4], dtype=np.float32)
                boxes[:, 0] = boxes[:, 0] / self.image_size[1]
                boxes[:, 1] = boxes[:, 1] / self.image_size[0]
                boxes[:, 2] = boxes[:, 2] / self.image_size[1]
                boxes[:, 3] = boxes[:, 3] / self.image_size[0]

                boxes = np.maximum(np.minimum(boxes, 1), 0)
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
                boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
                y = np.concatenate([boxes, y[:, -1:]], axis=-1)

            img = np.array(img, dtype=np.float32)
            #img.show()
            tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
            tmp_targets = np.array(y, dtype=np.float32)
        else:
            img, y = self.get_random_data_val(lines[index], self.image_size[0:2], False)
            if len(y) != 0:
                # 从坐标转换成0~1的百分比
                boxes = np.array(y[:, :4], dtype=np.float32)
                boxes[:, 0] = boxes[:, 0] / self.image_size[1]
                boxes[:, 1] = boxes[:, 1] / self.image_size[0]
                boxes[:, 2] = boxes[:, 2] / self.image_size[1]
                boxes[:, 3] = boxes[:, 3] / self.image_size[0]

                boxes = np.maximum(np.minimum(boxes, 1), 0)
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
                boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
                y = np.concatenate([boxes, y[:, -1:]], axis=-1)

            img = np.array(img, dtype=np.float32)

            tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
            tmp_targets = np.array(y, dtype=np.float32)
        
        return tmp_inp, tmp_targets


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes

