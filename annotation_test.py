#---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd

sets=[('train_relative'), ('val_relative'), ( 'test_relative')]

classes = ["fast-food-box","Cartons","battery","Cans","bottle","coat-hanger","cigarette"]

def convert_annotation(image_id, list_file):
    in_file = open('./data/VOCdevkit/Annotations_test/%s.xml'%(image_id), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
            
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for image_set in sets:
    image_ids = open('./data/VOCdevkit/ImageSets_test/%s.txt'%(image_set),encoding='utf-8').read().strip().split()
    list_file = open('./data/%s.txt'%(image_set), 'w',encoding='utf-8')
    for image_id in image_ids:
        list_file.write('%s/data/VOCdevkit/JPEGImages_test/%s.jpg'%(wd, image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()
