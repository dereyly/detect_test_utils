import os
import sys
import json
sys.path.insert(0,'/home/dereyly/git_progs/ns_982/DCN_SNMS')
from lib.dataset.dit import DIT
import json

# voc = json.load(open('/home/dereyly/ImageDB/VOCPascal/PASCAL_VOC/pascal_val2007.json','r'))
blist_type='test'
data_dir='/home/dereyly/ImageDB/DIT'
path_out='/home/dereyly/ImageDB/DIT/ann_json/%s.json' % blist_type
imdb = DIT(blist_type, 'data', data_dir)
gt_roidb = imdb.gt_roidb()
total = 0
count_im=0
count_ob=0
data={'annotations':[],'categories':[],'images':[]}
data['categories']=[{'supercategory': 'none', 'id': 1, 'name': 'person'}]
for recs in gt_roidb:
    count_im+=1
    #total += len(recs['boxes'])
    data['images'].append({'height':int(recs['height']),'width':int(recs['width']),'file_name':recs['image'],'id':count_im})
    for bb in recs['boxes']:
        count_ob+=1
        bb[[2,3]]-=bb[[0,1]]
        data['annotations'].append({'id':count_ob,'bbox':bb.tolist(),'segmentation':[],'area':0,'image_id':count_im,'category_id':1,'ignore':0,'iscrowd':0})
# print(total)
json.dump(data,open(path_out,'w'))
zz=0

