
# import caffe
import sys
sys.path.insert(0,'/home/dereyly/progs/pva-faster-rcnn-master/lib')
import numpy as np
import yaml
# from fast_rcnn.config import cfg
#from generate_anchors import generate_anchors
#from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps
import os
import pickle as pkl
import scipy.io as mat
import matplotlib.pyplot as plt
import argparse

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

def read_dataset(dir_meta,net_type='p-net'):
    data_out = {}
    data_name = 'boxes_' + net_type[0]
    is_mat = check_extension(dir_meta, 'mat')
    # prefix = 'WIDER_val/images/'
    if is_mat:
        if net_type == 'p-net':
            net_idx = 1
        if net_type == 'r-net':
            net_idx = 2
        if net_type == 'o-net':
            net_idx = 3
        for fname in os.listdir(dir_meta):
            path = dir_meta + fname
            if not os.path.isfile(path):
                continue
            with open(path, 'br') as f_in:
                data = mat.loadmat(f_in)
                # for key, val in data.items():
                if 'data_prop' in data:
                    data_o=data['data_prop']
                    for k, d in enumerate(data_o):
                        im_path = d[0][0]
                        key = im_path.split('/')[-2] + '/' + im_path.split('/')[-1]
                        # key=im_path
                        val = d[net_idx]
                        data_out[key] = val
                if 'data_out' in data:
                    data_o = data['data_out']
                    for k, d in enumerate(data_o):
                        im_path = d[0][0]
                        key = im_path.split('/')[-2] + '/' + im_path.split('/')[-1]
                        # key=im_path
                        val = d[1]
                        data_out[key] = val
    else:

        for fname in os.listdir(dir_meta):
            path = os.path.join(dir_meta,fname)
            if not os.path.isfile(path) or fname.split('.')[-1] != 'pkl':
                continue
            with open(path, 'br') as f_in:

                data = pkl.load(f_in)
                if data_name in data:
                    data=data[data_name]
                for key, val in data.items():
                    # key = prefix + key
                    data_out[key] = val
    return data_out

def evaluate(dir_detect_meta, dir_gt_meta, output_path, num_boxes_gt=None, net_type=None,visualize=False,step=0.001,precision_gt=None, th_iou=0.5):
    #load proposals
    output_path+='/'
    dir_detect_meta+='/'
    dir_gt_meta+='/'

    data_res=read_dataset(dir_detect_meta, net_type)
    data_all=read_dataset(dir_gt_meta)

    count=0
    n_props=0
    gt_all=0
    overlaps_all = np.array([]).reshape(0)
    props_all = np.array([]).reshape(0)
    props_ignored = np.array([]).reshape(0)
    for key, bb_props in data_res.items():
        # print(key)
        if not key in data_all:
            key_sp = key.split('/')
            key = key_sp[-2] + '/' + key_sp[-1]
        if not key in data_all:
            key='WIDER_val/images/'+key



        # print(list(data_all)[0])
        bb_gt = data_all[key]['bbox']
        if not isinstance(bb_props,dict): #todo tmp debug
            if len(bb_props)==2  and isinstance(bb_props[0],tuple):
                bb_props_convert=[]
                for bb in bb_props[1]:
                    bbc=list(bb[0])
                    bbc.append(bb[1])
                    bb_props_convert.append(bbc)
                bb_props=np.array(bb_props_convert)
                if len(bb_props)>0:
                    bb_props[:,[2,3]]=bb_props[:,[2,3]]+bb_props[:,[0,1]]

        if 'ignore_list' in data_all[key]:
            ignore_list=data_all[key]['ignore_list'].astype(bool)
        else:
            dim = len(data_all[key]['bbox'])
            ignore_list = np.zeros(dim,np.uint8)
            for n in range(dim):
                ignore_list[n]=np.array(data_all[key][n]['attributes']==0).any()

        bb_props=np.array(bb_props)
        #bb_props[:,[2,3]]=bb_props[:,[2,3]]-bb_props[:,[0,1]]
        w_gt = bb_gt[:, 2]
        bb_gt[:, [2, 3]] = bb_gt[:, [2, 3]] + bb_gt[:, [0, 1]]

        bb_gt_pos=bb_gt[ignore_list]
        bb_gt_ignored=bb_gt[np.logical_not(ignore_list)]

        pos_props_05_pos = calc_iou(bb_gt_pos,bb_props,thresh=th_iou)
        pos_props_05_ignored = calc_iou(bb_gt_ignored, bb_props,thresh=th_iou)

        overlaps_all = np.hstack((overlaps_all, pos_props_05_pos)) if pos_props_05_pos.size else overlaps_all
        props_all = np.hstack((props_all, bb_props[:, 4])) if bb_props.size else props_all
        props_ignored=np.hstack((props_ignored, pos_props_05_ignored)) if pos_props_05_ignored.size else props_ignored

        count+=1

        gt_all+=len(bb_gt_pos)
        #gt_all += (w>=30).sum()
        #print(pos_props_05_pos.shape[0],len(bb_gt_pos))

    del data_res
    del data_all
    print(gt_all,len(overlaps_all))
    recall=[]
    num_boxes=[]
    prec=[]
    k=0
    fname_out = output_path + 'roc/' + net_type + '.txt'
    fname_out_pkl = output_path + 'roc/' + net_type + '.pkl'
    if not os.path.exists(output_path + 'roc/'):
        os.makedirs(output_path + 'roc/')
    with open(fname_out,'w') as f_out:
        for th in np.arange(0,1,step):
            pos=(overlaps_all >th).sum()
            res=(props_all > th).sum()
            ignored=(props_ignored > th).sum()
            recall.append(pos/gt_all)
            res_pos=res-ignored
            if res_pos>0:

                num_boxes.append(res_pos/count)
                prec.append(pos/res_pos)
            else:
                num_boxes.append(0)
                prec.append(0)
            # print(th,recall[k],num_boxes[k],prec[k])
            print('th=%0.3f recall=%0.3f prec=%0.3f num_boxes=%d' % (th, recall[k], prec[k], num_boxes[k]))
            f_out.write('%0.3f %0.3f %d %0.4f' % (th,recall[k],num_boxes[k],prec[k]))
            k+=1
            zz=0
    prec=np.array(prec)
    recall=np.array(recall)
    num_boxes=np.array(num_boxes)

    AP = VOCap(recall, prec)
    print('AP=',AP)

    ths=np.arange(0,1,step)
    if not num_boxes_gt is None:
        idx=np.argmin(np.abs(num_boxes-num_boxes_gt))
        print('th=%0.3f recall=%0.3f num_boxes=%d' %(ths[idx],recall[idx],num_boxes[idx]))

    if not precision_gt is None:
        idx=np.argmin(np.abs(prec-precision_gt))
        print('th=%0.3f recall=%0.3f precision=%0.3f' % (ths[idx],recall[idx],prec[idx]))
    if visualize:
        plt.plot(recall, prec)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('ROC')
        plt.grid(True)
        plt.savefig(output_path+"roc_%s.png" % net_type)
        plt.show()
    pkl.dump({'prec':prec, 'recall':recall},open(fname_out_pkl,'wb'))

    return prec, recall
#not work -- /home/dereyly/ImageDB/mtcnn_data/12/mat/ /usr/local/images.git/WIDER/meta/ --dir_out=/home/dereyly/git_progs/facedet/ns/Seanlinx_mxnet_mtcnn/art_converted
#/home/dereyly/ImageDB/mtcnn_data/12/mat/ /tmp/wider_easy/ --dir_out=/home/dereyly/git_progs/facedet/ns/Seanlinx_mxnet_mtcnn/art_converted
#/home/dereyly/ImageDB/mtcnn_data/mxnet_pnet/meta_pretrained/ /home/dereyly/git_progs/facedet/ns/Seanlinx_mxnet_mtcnn/art_converted/gt/wider_easy --dir_out=/home/dereyly/git_progs/facedet/ns/Seanlinx_mxnet_mtcnn/art_converted --net_type=o-net
#/home/dereyly/git_progs/facedet/ns/Seanlinx_mxnet_mtcnn/art_converted/gt/wider_easy /usr/local/images.git/WIDER/meta/  --dir_out=/home/dereyly/git_progs/facedet/ns/Seanlinx_mxnet_mtcnn/art_converted
#/home/dereyly/ImageDB/mtcnn_data/mxnet_pnet/meta_hard/ /home/dereyly/git_progs/facedet/ns/Seanlinx_mxnet_mtcnn/art_converted/gt/wider_easy --dir_out=/tmp/pred --net_type=p-net
#/home/dereyly/ImageDB/mtcnn_data/all4/mxnet_onet/meta_test /home/dereyly/git_progs/facedet/ns/Seanlinx_mxnet_mtcnn/art_converted/gt/wider_easy --dir_out=/tmp/pred --net_type=o-net

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate script MTCNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir_detect_meta',  type=str)
    parser.add_argument('dir_gt_meta', type=str)
    parser.add_argument('--dir_out', dest='dir_out', help='output data folder', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='data/fddb', type=str)
    parser.add_argument('--net_type', dest='net_type', help='type of network: p-net | r-net | o-net',
                        default='p-net', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with argument:')
    print(args)

    #min_face = 20
    num_boxes_gt=None
    if args.net_type == 'p-net':
        num_boxes_gt = 522

    if args.net_type == 'r-net':
        num_boxes_gt = 61
    precision_gt=None
    if args.net_type == 'o-net':
        precision_gt =0.915 #it is mtcnn resault with th=0.9


    evaluate(dir_detect_meta=args.dir_detect_meta, dir_gt_meta=args.dir_gt_meta, output_path=args.dir_out,
             num_boxes_gt=num_boxes_gt, precision_gt=precision_gt,visualize=False, net_type=args.net_type, th_iou=0.5)