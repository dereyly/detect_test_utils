import os
import sys
import json
try:
    from tools.read_voc import VOCReader
except:
    current_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0,current_path+'/tools')
    from read_voc import VOCReader
import json
import cPickle as pkl
#import pickle as pkl
import numpy as np
#data_res=pkl.load(open('/home/dereyly/progs/Detectron/test/voc_dit_test/generalized_rcnn/detections.pkl','rb'),encoding='latin1')
# data_res=pkl.load(open('/home/dereyly/progs/Detectron/test/voc_dit_test/generalized_rcnn/detections.pkl','rb'))
# voc = json.load(open('/home/dereyly/ImageDB/VOCPascal/PASCAL_VOC/pascal_val2007.json','r'))
#data_dir='/home/dereyly/ImageDB/VOCPascal/VOCdevkit/VOC2007/'
data_name='trainval'
data_dir = '/home/dereyly/ImageDB/food/VOC5180/'
content = data_dir+'/ImageSets/Main/%s.txt' % data_name
ann_dir=data_dir+'/Annotations/'
img_dir=data_dir+'/JPEGImages/'

vread=VOCReader(data_name='food')
data={}
with open(content,'r') as f_in:
    for line in f_in.readlines():
        line=line.strip()
        #ext=content.split('.')[-1]
        path=ann_dir+line+'.xml'
        data[line]=vread._load_pascal_annotation(path)
pkl.dump(data,open(data_dir+data_name+'_gt.pkl','wb'))


