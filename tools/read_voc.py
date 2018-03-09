import numpy as np
import xml.etree.ElementTree as ET
import scipy.sparse
import os

def get_cls2ctg():
    fname='/home/dereyly/ImageDB/food/VOC5180/sku_space_package_type_list.txt'
    cls2ctg={}
    with open(fname,'r') as f:
        for line in f.readlines():
            line=line.strip()
            sl = line.split(' ')
            cls2ctg[sl[0]]=sl[1]
    return cls2ctg

class VOCReader():
    def __init__(self, data_name='pascal'):
        if data_name=='pascal':
            self._classes = ('__background__',  # always index 0
                             'aeroplane', 'bicycle', 'bird', 'boat',
                             'bottle', 'bus', 'car', 'cat', 'chair',
                             'cow', 'diningtable', 'dog', 'horse',
                             'motorbike', 'person', 'pottedplant',
                             'sheep', 'sofa', 'train', 'tvmonitor')
            self.num_classes = len(self._classes)
            self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        elif data_name=='food':
            self._classes = ('__background__',  # always index 0
                             'mcm', 'cbb', 'gb', 'tu', 'vcm', 'fc', 'bo', 'ac', 'glm', 'lp',
                             'pbm', 'ob', 'ct', 'dp', 'flp', 'cb', 'mc',
                             'fo', 'vac', 'cbl', 'bo', 'gl')


            cls2ctg = get_cls2ctg()
            self.num_classes = len(self._classes)
            class_to_ind_loc = dict(zip(self._classes, range(self.num_classes)))
            self._class_to_ind = {}
            for key,val in cls2ctg.iteritems():
                self._class_to_ind[key]=class_to_ind_loc[val]
        else:
            raise ValueError('No dataset ' +data_name)

        self.is_difficult=False
    def _load_pascal_annotation(self, filename, is_difficult=False):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        # filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not is_difficult:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(1, float(bbox.find('xmin').text) - 1)
            y1 = max(1, float(bbox.find('ymin').text) - 1)
            x2 = max(1, float(bbox.find('xmax').text) - 1)
            y2 = max(1, float(bbox.find('ymax').text) - 1)
            if x1 > 15000 or x2 > 15000 or y1 > 15000 or y2 > 15000 or x1 < 0 or y1 < 0:
                print('errrr')
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}