import os
import tqdm
import pandas as pd


print('\n=== Create Annotation File ===\n')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
META_DATA_DIR = os.path.join(DATA_DIR, 'meta_data')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_img')
TRAIN_MASK_IMG_DIR = os.path.join(DATA_DIR, 'train_mask_img')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'val_img')
VAL_MASK_IMG_DIR = os.path.join(DATA_DIR, 'val_mask_img')
CLASS_DESCRIPTION_PATH = os.path.join(META_DATA_DIR, 'challenge-2019-classes-description-segmentable.csv')
ORIGIN_TRAIN_MASK_ANNOTATION_FILE_PATH = os.path.join(META_DATA_DIR, 'challenge-2019-train-segmentation-masks.csv')
ORIGIN_VAL_MASK_ANNOTATION_FILE_PATH = os.path.join(META_DATA_DIR, 'challenge-2019-validation-segmentation-masks.csv')
TRAIN_ANNOTATION_FILE_PATH = os.path.join(META_DATA_DIR, 'train_annotation.txt')
VAL_ANNOTATION_FILE_PATH = os.path.join(META_DATA_DIR, 'validation_annotation.txt')

df_origin_train_mask_annotation = pd.read_csv(ORIGIN_TRAIN_MASK_ANNOTATION_FILE_PATH).sort_values(['ImageID']).reset_index()
df_origin_val_mask_annotation = pd.read_csv(ORIGIN_VAL_MASK_ANNOTATION_FILE_PATH).sort_values(['ImageID']).reset_index()
df_class_descriptions = pd.read_csv(CLASS_DESCRIPTION_PATH, names=['LabelName', 'ClassName'])
label_mapping_to_class_name = dict(zip(df_class_descriptions['LabelName'], df_class_descriptions['ClassName']))


# Create Annotation File Function
def create_annotation_file(create_mode):
    if create_mode == 'train':
        IMG_DIR = TRAIN_IMG_DIR
        MASK_IMG_DIR = TRAIN_MASK_IMG_DIR
        DF_MASK_ANNOTATION = df_origin_train_mask_annotation
        ANNOTATION_FILE_SAVE_PATH = TRAIN_ANNOTATION_FILE_PATH
    elif create_mode == 'validation':
        IMG_DIR = VAL_IMG_DIR
        MASK_IMG_DIR = VAL_MASK_IMG_DIR
        DF_MASK_ANNOTATION = df_origin_val_mask_annotation
        ANNOTATION_FILE_SAVE_PATH = VAL_ANNOTATION_FILE_PATH
    else:
        print("create_mode: 'train' or 'validation'")
        return

    f = open(ANNOTATION_FILE_SAVE_PATH, 'wt')

    for idx, row in tqdm.tqdm(
            DF_MASK_ANNOTATION.iterrows(),
            desc=f'Create {create_mode} annotation file process',
            total=DF_MASK_ANNOTATION.shape[0],
    ):
        img_id = row['ImageID']
        img_file_path = os.path.join(IMG_DIR, img_id + '.jpg')
        mask_img_file_path = os.path.join(MASK_IMG_DIR, row['MaskPath'])
        label_name = row['LabelName']
        class_name = label_mapping_to_class_name[label_name]

        cur_row = img_id + ',' + \
                img_file_path + ',' + \
                mask_img_file_path + ',' + \
                class_name + '\n'
        f.write(cur_row)

    f.close()
    print()


# Create Annotation File
create_annotation_file(create_mode='train')
create_annotation_file(create_mode='validation')
print('\n === Complete Create Annotation File ===')
