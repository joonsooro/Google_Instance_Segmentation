import os
import sys
import tqdm
import numpy as np
import cv2
from sub_func.get_data import get_data

# Add Path For Loading mrcnn
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils


__all__ = [
    'ROOT_DIR', 'DATA_DIR', 'META_DATA_DIR',
    'TRAIN_IMG_DIR', 'TRAIN_MASK_IMG_DIR',
    'VAL_IMG_DIR', 'VAL_MASK_IMG_DIR',
    'TEST_IMG_DIR',
    'MODEL_DIR', 'SAVED_MODEL_DIR', 'PRETAINED_MODEL_DIR', 'SUBMISSION_DIR',
    'TRAIN_ANNOTATION_FILE_PATH', 'VAL_ANNOTATION_FILE_PATH', 'COCO_MODEL_PATH',
    'CLASS_DESCRIPTION_PATH', 'EMPTY_SUBMISSION_PATH', 'SUBMISSION_PATH', 'DIRECT_MODEL_PATH',
    'TRAIN_IMG_INFO_LIST', 'TRAIN_CLASS_CNT', 'TRAIN_CLASS_MAPPING',
    'VAL_IMG_INFO_LIST', 'VAL_CLASS_CNT',
    'OpenImgConfig', 'OpenImgDataSet',
    'TRAIN_NB_IMG', 'TRAIN_NB_CLASS',
    'CONFIG', 'TRAIN_DATASET', 'VAL_DATASET',
    'TRAIN_INIT_WITH', 'EPOCHS',
]

DATA_DIR = os.path.join(ROOT_DIR, 'data')
META_DATA_DIR = os.path.join(DATA_DIR, 'meta_data')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_img')
TRAIN_MASK_IMG_DIR = os.path.join(DATA_DIR, 'train_mask_img')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'val_img')
VAL_MASK_IMG_DIR = os.path.join(DATA_DIR, 'val_mask_img')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test_img')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)
SAVED_MODEL_DIR = os.path.join(MODEL_DIR, 'saved_model')
if not os.path.exists(SAVED_MODEL_DIR):
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
PRETAINED_MODEL_DIR = os.path.join(ROOT_DIR, 'pretained_model')
if not os.path.exists(PRETAINED_MODEL_DIR):
    os.makedirs(PRETAINED_MODEL_DIR, exist_ok=True)
SUBMISSION_DIR = os.path.join(ROOT_DIR, 'submission')
if not os.path.exists(SUBMISSION_DIR):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_ANNOTATION_FILE_PATH = os.path.join(META_DATA_DIR, 'train_annotation.txt')
VAL_ANNOTATION_FILE_PATH = os.path.join(META_DATA_DIR, 'validation_annotation.txt')
COCO_MODEL_PATH = os.path.join(PRETAINED_MODEL_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

CLASS_DESCRIPTION_PATH = os.path.join(META_DATA_DIR, 'challenge-2019-classes-description-segmentable.csv')
EMPTY_SUBMISSION_PATH = os.path.join(META_DATA_DIR, 'sample_empty_submission.csv')
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, 'submission.csv')

# Get Basic Data From Train Data Annotation File
TRAIN_IMG_INFO_LIST, TRAIN_CLASS_INFO_LIST, TRAIN_CLASS_CNT, TRAIN_CLASS_MAPPING = get_data(TRAIN_ANNOTATION_FILE_PATH)
TRAIN_NB_IMG = len(TRAIN_IMG_INFO_LIST)
TRAIN_NB_CLASS = len(TRAIN_CLASS_MAPPING)

# Get Basic Data From Validation Data Annotation File
VAL_IMG_INFO_LIST, VAL_CLASS_INFO_LIST, VAL_CLASS_CNT, _ = get_data(VAL_ANNOTATION_FILE_PATH)
VAL_NB_IMG = len(VAL_IMG_INFO_LIST)

# Train Settings
TRAIN_INIT_WITH = "coco"  # imagenet, coco, or direct_set
DIRECT_MODEL_NAME = 'Your saved model name'  # Set model file name if you use TRAIN_INIT_WITH == 'direct_set'
DIRECT_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, DIRECT_MODEL_NAME)
EPOCHS = 1

# Define Get Step Function
def get_steps(num_data, batch_size):
    quotient, remainder = divmod(num_data, batch_size)
    return (quotient + 1) if remainder else quotient


class OpenImgConfig(Config):
    NAME = "Open_Images_Segmentation_2019"
    NUM_CLASSES = TRAIN_NB_CLASS  # Number of classes (including background)
    LEARNING_RATE = 1e-6
    GPU_COUNT = 1  # Default: 1
    IMAGES_PER_GPU = 2  # Default: 2
    BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
    STEPS_PER_EPOCH = get_steps(TRAIN_NB_IMG, BATCH_SIZE)  # Default: 1000
    VALIDATION_STEPS = get_steps(VAL_NB_IMG, BATCH_SIZE)  # Default: 50

    USE_MINI_MASK = True  # Default: True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask [Default: (56, 56)]
    IMAGE_RESIZE_MODE = "square"  # Default: "square"
    IMAGE_MIN_DIM = int(800 * (3/4))  # Default: 800
    IMAGE_MAX_DIM = int(1024 * (3/4))  # Default: 1024


class OpenImgDataSet(utils.Dataset):
    def load_mask(self, image_id):
        mask_data_list = self.image_info[image_id]['masks']
        class_ids = list()
        masks = list()
        for mask_data in mask_data_list:
            mask = cv2.imread(mask_data['mask_img_path'])
            mask = mask[:, :, 0]  # Remove Channel Dimension
            masks.append(mask)
            class_ids.append(mask_data['class_id'])

        masks_arr = np.stack(masks, axis=2).astype(np.bool)  # (height, width, instance count)
        class_ids_arr = np.array(class_ids).astype(np.int32)
        return masks_arr, class_ids_arr

    def prepare_modified(self, class_map, mode):
        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(class_map)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = list(class_map.keys())
        self.class_names[0] = 'BG'  # 'bg' to 'BG'
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in tqdm.tqdm(self.sources, desc=f'{mode} Dataset Prepare', total=len(self.sources)):
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)


CONFIG = OpenImgConfig()
TRAIN_DATASET = OpenImgDataSet()
VAL_DATASET = OpenImgDataSet()

# Add Train Data To Train DataSet
for idx, img_info in tqdm.tqdm(
        enumerate(TRAIN_IMG_INFO_LIST),
        desc='Train Dataset (add_image)',
        total=len(TRAIN_IMG_INFO_LIST)
):
    TRAIN_DATASET.add_image(
        image_id=idx,
        source='open_images_dataset_v5_train',
        path=img_info['img_path'],
        masks=img_info['masks'],
    )

for idx, cls_info in tqdm.tqdm(
        enumerate(TRAIN_CLASS_INFO_LIST),
        desc='Train Dataset (add_class)',
        total=len(TRAIN_CLASS_INFO_LIST),
):
    TRAIN_DATASET.add_class(
        source='open_images_dataset_v5_train',
        class_id=cls_info['class_id'],
        class_name=cls_info['class_name'],
    )

# Add Validation Data To Validation DataSet
for idx, img_info in tqdm.tqdm(
        enumerate(VAL_IMG_INFO_LIST),
        desc='Validation Dataset (add_image)',
        total=len(VAL_IMG_INFO_LIST),
):
    VAL_DATASET.add_image(
        image_id=idx,
        source='open_images_dataset_v5_validation',
        path=img_info['img_path'],
        masks=img_info['masks'],
    )

for idx, cls_info in tqdm.tqdm(
        enumerate(VAL_CLASS_INFO_LIST),
        desc='Validation Dataset (add_class)',
        total=len(VAL_CLASS_INFO_LIST),
):
    VAL_DATASET.add_class(
        source='open_images_dataset_v5_validation',
        class_id=cls_info['class_id'],
        class_name=cls_info['class_name'],
    )
