import os
import sys
import json
import cv2
im_dir = '/home/dereyly/ImageDB/food/VOC5180/JPEGImages/'
ext='jpg' #'JPG'#train #TODO make loop for extentions
python_version = sys.version_info.major
if python_version == 3:
    import pickle as pkl
else:
    import cPickle as pkl

import json

blist_type ='test'

data_dir = '/home/dereyly/ImageDB/food/VOC5180/'
path_pkl=data_dir+blist_type+'_gt.pkl'
path_out=data_dir+'/ann_json/%s.json' % blist_type
if not os.path.exists(data_dir+'/ann_json'):
    os.makedirs(data_dir+'/ann_json')

data_gt=pkl.load(open(path_pkl,'rb'))

total = 0
count_im=0
count_ob=0
data={'annotations':[],'categories':[],'images':[]}
data['categories']=[]
classes = ('__background__',  # always index 0
                 'mcm', 'cbb', 'gb', 'tu', 'vcm', 'fc', 'bo', 'ac', 'glm', 'lp',
                 'pbm', 'ob', 'ct', 'dp', 'flp', 'cb', 'mc',
                 'fo', 'vac', 'cbl', 'bo', 'gl')
for k,cls in enumerate(classes):
    data['categories'].append({'supercategory': 'none', 'id': k, 'name': cls})
for key, val in data_gt.iteritems(): #python2
    count_im+=1
    # if count_im>100:
    #     break
    #total += len(recs['boxes'])
    if key.split('.')[-1]!=ext:
        key+='.'+ext
    path=im_dir+key
    im=cv2.imread(path)
    if im is None:
        continue
    sz=im.shape
    data['images'].append({'height':sz[0],'width':sz[1],'file_name':key,'id':count_im})
    for k,bb in enumerate(val['boxes']):
        count_ob+=1
        bb[[2,3]]-=bb[[0,1]]
        data['annotations'].append({'id':count_ob,'bbox':bb.tolist(),'segmentation':[],'area':int(bb[2]*bb[3]),'image_id':count_im,'category_id':int(val['gt_classes'][k]),'ignore':0,'iscrowd':0})
    #break
# print(total)
json.dump(data,open(path_out,'w'))
zz=0

