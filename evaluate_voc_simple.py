
# import caffe
import sys
sys.path.insert(0,'/home/dereyly/progs/pva-faster-rcnn/lib')
import numpy as np
import yaml
# from fast_rcnn.config import cfg
#from generate_anchors import generate_anchors
#from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps
import os
#import pickle as pkl
import cPickle as pkl
import scipy.io as mat
import matplotlib.pyplot as plt
import argparse
import cv2


img_dir='/home/dereyly/ImageDB/food/VOC5180/JPEGImages/'

def VOCap(prec, rec): #copy this function from matlab
    mrec=np.hstack((0,rec,1))
    mpre=np.hstack((0,prec,0))

    for i in reversed(range(len(mpre)-1)):
        mpre[i]=max(mpre[i],mpre[i+1])

    idx=np.where(mrec[1:]!=mrec[-1])[0]+1
    ap=np.sum((mrec[idx]-mrec[idx-1])*mpre[idx])
    return ap

def calc_iou(gt_boxes,proposals,thresh, return_index = False):
    #print proposals.shape[0], gt_boxes
    try:
        overlaps = bbox_overlaps(
            np.ascontiguousarray(proposals, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)

        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        intersect_05=gt_argmax_overlaps[gt_max_overlaps>thresh]

    except:
        return np.array([]).reshape(0)

    if return_index:
        #gt_mask = gt_max_overlaps > thresh
        return max_overlaps,gt_max_overlaps, gt_assignment
    if len(intersect_05)==0:
        return np.array([]).reshape(0)
    else:
        return proposals[intersect_05,4]

def check_extension(dir_in, ext_check='mat'):
    #ToDo maybe return all extensions
    for fname in os.listdir(dir_in):
        path = dir_in + fname
        if not os.path.isfile(path):
            continue
        ext=fname.split('.')[-1]
        if ext==ext_check:
            return True
    return False

def read_meta(dir_meta, data_name=''):
    data_out={}
    for fname in os.listdir(dir_meta):
        path = os.path.join(dir_meta, fname)
        if not os.path.isfile(path) or fname.split('.')[-1] != 'pkl':
            continue
        with open(path, 'rb') as f_in:
            try:
                data = pkl.load(f_in)
            except:
                data = pkl.load(f_in, encoding = 'latin1')
            if data_name in data:
                data = data[data_name]
            for key, val in data.items():
                # key = prefix + key
                data_out[key] = val
    return data_out

def evaluate(dir_detect_meta, path_gt_meta,visualize=False,step=0.001,num_cls=2, NMS=0.3):
    #load proposals

    # pkl.load(open(path_detect_meta,'rb'),encoding='latin1')
    data_res=read_meta(dir_detect_meta)
    data_all=pkl.load(open(path_gt_meta,'rb'))
    multi_cls = num_cls>2
    if multi_cls:
        APs=np.zeros(num_cls)

    mAP=0
    for j in range(1,num_cls):
        count=0
        n_props=0
        gt_all=0
        overlaps_all = np.array([]).reshape(0)
        props_all = np.array([]).reshape(0)
        props_ignored = np.array([]).reshape(0)

        for key, bb_props in data_res.items():
            # print(key)
            #bb_props=np.array(bb_props[1])
            bb_props = bb_props['bbox']
            if not key in data_all:
                key_sp = key.split('/')
                key = key_sp[-2] + '/' + key_sp[-1]
            if not key in data_all:
                key='WIDER_val/images/'+key



            # print(list(data_all)[0])
            bb_gt = data_all[key]['boxes'].astype(float) #['bbox']
            if multi_cls:
                cls=data_all[key]['gt_classes']
                bb_gt=bb_gt[cls==j]
                bb_props=bb_props[bb_props[:,5]==j]
            else:
                keep = nms(bb_props[:,:5].astype(np.float32), NMS)
                bb_props = bb_props[keep]

            if len(bb_gt)==0:
                continue
            # if not isinstance(bb_props,dict): #todo tmp debug
            #     if len(bb_props)==2  and isinstance(bb_props[0],tuple):
            #         bb_props_convert=[]
            #         for bb in bb_props[1]:
            #             bbc=list(bb[0])
            #             bbc.append(bb[1])
            #             bb_props_convert.append(bbc)
            #         bb_props=np.array(bb_props_convert)
            #         if len(bb_props)>0:
            #             bb_props[:,[2,3]]=bb_props[:,[2,3]]+bb_props[:,[0,1]]

            # if 'ignore_list' in data_all[key]:
            #     ignore_list=data_all[key]['ignore_list'].astype(bool)
            # else:
            #     dim = len(data_all[key]['bbox'])
            #     ignore_list = np.zeros(dim,np.uint8)
            #     for n in range(dim):
            #         ignore_list[n]=np.array(data_all[key][n]['attributes']==0).any()
            # ignore_list=np.ones(bb_gt.shape,int)
            bb_props=np.array(bb_props)
            # bb_props = bb_props[bb_props[:, 4] > 0.001]
            #bb_props[:,[2,3]]=bb_props[:,[2,3]]-bb_props[:,[0,1]]
            # w_gt = bb_gt[:, 2]
            # bb_gt[:, [2, 3]] = bb_gt[:, [2, 3]] + bb_gt[:, [0, 1]]

            # bb_gt_pos=bb_gt[ignore_list]
            # bb_gt_ignored=bb_gt[np.logical_not(ignore_list)]
            bb_gt_pos=bb_gt
            bb_gt_ignored=np.array([])
            pos_props_05_pos = calc_iou(bb_gt_pos,bb_props,thresh=0.5)
            pos_props_05_ignored = calc_iou(bb_gt_ignored, bb_props,thresh=0.5)

            overlaps_all = np.hstack((overlaps_all, pos_props_05_pos)) if pos_props_05_pos.size else overlaps_all
            props_all = np.hstack((props_all, bb_props[:, 4])) if bb_props.size else props_all
            props_ignored=np.hstack((props_ignored, pos_props_05_ignored)) if pos_props_05_ignored.size else props_ignored

            count+=1
            gt_all += bb_gt.shape[0] #(w_gt >= 30).sum()
            if visualize and 0:
                img=cv2.imread(img_dir+key+'.jpg')
                for bb in bb_gt:
                    bb=bb.astype(int)
                    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color=(0, 0, 255),thickness=5)
                if len(bb_props)>0:
                    bb_props_filt=bb_props[bb_props[:,4]>0.3]
                    for bb in bb_props_filt:
                        bb = bb.astype(int)
                        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color=(0, 255, 0), thickness=3)
                    cv2.imwrite('/home/dereyly/ImageDB/food/VOC5180/img.jpg', img)
                    #cv2.imshow('img', img)
                    #cv2.waitKey(0)
                zz=0
            #
            #print(pos_props_05_pos.shape[0],len(bb_gt_pos))

        # del data_res
        # del data_all
        print(gt_all,len(overlaps_all))
        recall=[]
        num_boxes=[]
        prec=[]
        k=0


        for th in np.arange(0,1,step):
            pos=(overlaps_all >th).sum().astype(float)
            res=(props_all > th).sum().astype(float)
            ignored=(props_ignored > th).sum().astype(float)
            recall.append(pos/gt_all)
            res_pos=res-ignored
            if res_pos>0:

                num_boxes.append(res_pos/count)
                prec.append(pos/res_pos)
            else:
                num_boxes.append(0)
                prec.append(1)
            if not multi_cls:
                print('th=%0.3f recall=%0.3f prec=%0.3f num_boxes=%d' % (th, recall[k], prec[k], num_boxes[k]))
            k+=1
            zz=0


        prec=np.array(prec)
        recall=np.array(recall)
        num_boxes=np.array(num_boxes)

        AP = VOCap(recall, prec)
        print('AP=',AP)
        #mAP+=AP
        if multi_cls:
            APs[j-1]=AP
    if multi_cls:
        mAP=APs[APs>0.1].mean()
        print(mAP)
    # ths=np.arange(0,1,step)
    # if not num_boxes_gt is None:
    #     idx=np.argmin(np.abs(num_boxes-num_boxes_gt))
    #     print('th=%0.3f recall=%0.3f num_boxes=%d' %(ths[idx],recall[idx],num_boxes[idx]))
    #
    # if not precision_gt is None:
    #     idx=np.argmin(np.abs(prec-precision_gt))
    #     print('th=%0.3f recall=%0.3f precision=%0.3f' % (ths[idx],recall[idx],prec[idx]))
    if visualize:
        plt.plot(recall, prec)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('ROC')
        plt.grid(True)
        plt.show()

    return prec, recall

#/home/dereyly/ImageDB/mtcnn_data/all4/mxnet_onet/meta_test /home/dereyly/git_progs/facedet/ns/Seanlinx_mxnet_mtcnn/art_converted/gt/wider_easy --dir_out=/tmp/pred --net_type=o-net
#/home/dereyly/progs/Detectron/test/voc_dit_test/generalized_rcnn/detections_dict.pkl /home/dereyly/ImageDB/DIT/ann_json/test.pkl
#/home/dereyly/ImageDB/VOCPascal/VOCdevkit/VOC2007/res /home/dereyly/ImageDB/VOCPascal/VOCdevkit/VOC2007/gt.pkl
#/home/dereyly/ImageDB/food/VOC5180/res /home/dereyly/ImageDB/food/VOC5180/test_gt.pkl
def parse_args():
    parser = argparse.ArgumentParser(description='evaluate script MTCNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path_detect_meta',  type=str)
    parser.add_argument('path_gt_meta', type=str)
    parser.add_argument('--dir_out', dest='dir_out', help='output data folder', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='data/fddb', type=str)
    parser.add_argument('--num_cls', dest='num_cls',
                        default='2', type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with argument:')
    print(args)


    evaluate(dir_detect_meta=args.path_detect_meta, path_gt_meta=args.path_gt_meta,visualize=True,num_cls=args.num_cls)