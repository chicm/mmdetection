import os.path as osp

ROOT_DIR = '/home/chec/data/open-images'

SEG_DATA_DIR = osp.join(ROOT_DIR, 'segmentation')
REL_DATA_DIR = osp.join(ROOT_DIR, 'relation')
DET_DATA_DIR = osp.join(ROOT_DIR, 'detect')
IMG_DIR = osp.join(ROOT_DIR, 'train', 'imgs')
MASK_DIR = osp.join(ROOT_DIR, 'masks', 'train')
TEST_IMG_DIR = osp.join(ROOT_DIR, 'test')
VAL_IMG_DIR = osp.join(ROOT_DIR, 'val')
