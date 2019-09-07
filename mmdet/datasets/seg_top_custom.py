import os.path as osp
import pandas as pd
import pickle
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
from sklearn.utils import shuffle
#from detect.utils import get_image_size
from multiprocessing import Pool
from tqdm import tqdm
import struct
import imghdr
import cv2
import glob
from .settings import SEG_DATA_DIR as DATA_DIR, IMG_DIR, MASK_DIR, TEST_IMG_DIR, VAL_IMG_DIR


def get_top_classes(start_index=0, end_index=275):
    df = pd.read_csv(osp.join(DATA_DIR, 'top_classes_level1.csv'))
    c = df['class'].values[start_index:end_index]
    #print(df.head())
    stoi = { c[i]: i for i in range(len(c)) }
    return c, stoi

#classes, stoi = get_top_classes(0, 276)
classes, stoi = None, None
#print(classes)

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            raise AssertionError('imghead len != 24')
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                raise AssertionError('png check failed')
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                raise
        else:
            print(fname, imghdr.what(fname))
            #raise AssertionError('file format not supported')
            img = cv2.imread(fname)
            print(img.shape)
            height, width, _ = img.shape

        return width, height

def get_mask(mask_fn, img_sz):
    mask = cv2.imread(osp.join(MASK_DIR, mask_fn))
    mask = cv2.resize(mask, img_sz)
    #print(mask.shape)
    return mask[:,:,0] / 255

def group2mmdetection(group: dict) -> dict:
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    image_id, group = group
    filename = group['filename'].values[0]
    fullpath = osp.join(IMG_DIR, filename)
    assert image_id == osp.basename(filename).split('.')[0]

    width, height = get_image_size(fullpath)

    group['BoxXMin'] = group['BoxXMin'] * width
    group['BoxXMax'] = group['BoxXMax'] * width
    group['BoxYMin'] = group['BoxYMin'] * height
    group['BoxYMax'] = group['BoxYMax'] * height

    bboxes = [np.expand_dims(group[col].values, -1) for col in['BoxXMin', 'BoxYMin', 'BoxXMax', 'BoxYMax']]
    bboxes = np.concatenate(bboxes, axis=1)
    #print(bboxes)
    #print(bboxes.shape)
    return {
        'filename': group['filename'].values[0], #image_id+'.jpg',
        'width': width,
        'height': height,
        'ann': {
            'bboxes': np.array(bboxes, dtype=np.float32),
            'labels': np.array([stoi[x] for x in group['LabelName'].values]) + 1,
            'masks': group['MaskPath'].values
        }
    }

def get_balanced_meta(start_index, end_index):
    global classes, stoi
    print('start_index:', start_index, 'end_index:', end_index)
    classes, stoi = get_top_classes(start_index, end_index)
    df_masks = pd.read_csv(osp.join(DATA_DIR, 'challenge-2019-train-segmentation-masks.csv'))

    #classes_0_20, _ = get_top_classes(0, 20)
    #classes_20_100, _ = get_top_classes(20, 100)
    #classes_100_276, _ = get_top_classes(100, 276)

    #imgs_100_276 = set(df_masks.loc[df_masks.LabelName.isin(set(classes_100_276))].ImageID.unique())
    #imgs_20_100 = set(df_masks.loc[df_masks.LabelName.isin(set(classes_20_100))].ImageID.unique()) - imgs_100_276
    #imgs_0_20 = set(df_masks.loc[df_masks.LabelName.isin(set(classes_0_20))].ImageID.unique()) - imgs_20_100 - imgs_100_276
    #print(len(imgs_0_20), len(imgs_20_100), len(imgs_100_276))

    #selected_imgs = list(imgs_100_276) + shuffle(list(imgs_20_100))[:40000] + shuffle(list(imgs_0_20))[:20000] #[:1000]
    #df_masks = df_masks.loc[df_masks.ImageID.isin(set(selected_imgs))]
    meta = df_masks.loc[df_masks.LabelName.isin(set(classes))]
    
    print(meta.shape)
    print('num images:', len(meta.ImageID.unique()))
    #print('num images:', len(selected_imgs))

    img_files = glob.glob(IMG_DIR + '/**/*.jpg')
    fullpath_dict = {}
    for fn in img_files:
        fullpath_dict[osp.basename(fn).split('.')[0]] = osp.join(osp.basename(osp.dirname(fn)), osp.basename(fn))
    
    meta['filename'] = meta.ImageID.map(lambda x: fullpath_dict[x])

    return meta

#def get_val_meta():
#    df_box = pd.read_csv(osp.join(DATA_DIR, 'challenge-2019-validation-vrd-bbox.csv'))
#    df_box['filename'] = df_box.ImageID.map(lambda x: x+'.jpg')

def id2mmdetection(img_id):
    fn = osp.join(TEST_IMG_DIR, '{}.jpg'.format(img_id))
    width, height = get_image_size(fn)
    return {
        'filename': img_id+'.jpg',
        'width': width,
        'height': height,
    }

def get_test_ds():
    df = pd.read_csv(osp.join(DATA_DIR, 'sample_empty_submission.csv')) #.iloc[:5000]
    with Pool(50) as p:
        img_ids = df.ImageID.values
        annos = list(tqdm(iterable=p.map(id2mmdetection, img_ids), total=len(img_ids)))
    print(annos[0])
    print('DATASET LEN:', len(annos))
    return annos

#def id2mmdetection_val(img_id):
#    fn = osp.join(VAL_IMG_DIR, '{}.jpg'.format(img_id))
#    width, height = get_image_size(fn)
#    return {
#        'filename': img_id+'.jpg',
#        'width': width,
#        'height': height,
#    }

#def get_val_ds():
#    df = pd.read_csv(osp.join(DATA_DIR, 'val_imgs.csv'))
#    with Pool(50) as p:
#        img_ids = df.ImageId.values
#        annos = list(tqdm(iterable=p.map(id2mmdetection_val, img_ids), total=len(img_ids)))
#    print(annos[0])
#    print('DATASET LEN:', len(annos))
#    return annos

class SegTopCustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 start_index=None,
                 end_index=None):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(ann_file, start_index, end_index)
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.img_infos)

    def load_train_annotations(self, start_index, end_index):
        meta = get_balanced_meta(start_index, end_index)
        print('grouping...')
        groups = list(meta.groupby('ImageID'))

        with Pool(50) as p:
            annos = list(tqdm(iterable=p.imap_unordered(group2mmdetection, groups), total=len(groups)))

        print('DATASET LEN:', len(annos))
        print('ann:', annos[:2])
    
        return shuffle(annos)

    def load_annotations(self, ann_file, start_index, end_index):
        if 'train' in ann_file:
            return self.load_train_annotations(start_index, end_index)
        elif 'test' in ann_file:
            return get_test_ds()
        elif 'val' in ann_file:
            return get_val_ds()

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            img_sz = (img_info['width'], img_info['height'])
            masks = [get_mask(x, img_sz) for x in ann['masks']]
            #print(masks[0].shape, pad_shape, img_shape)
            gt_masks = self.mask_transform(masks, pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
