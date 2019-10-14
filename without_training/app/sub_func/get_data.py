import os
import tqdm
import pickle
import cv2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, 'train_class_mapping.pickle')


def get_data(annotation_file_path):
    cls_cnt = dict()
    img_info = dict()
    cls_info_list = list()
    # cls_mapping = dict(BG=0)
    with open(CLASS_MAPPING_PATH, 'rb') as f:
        cls_mapping = pickle.load(f)

    print('==== Start Get Data From Annotation File ====')
    line_length = len(open(annotation_file_path, 'rt').readlines())
    with open(annotation_file_path, 'rt') as f:
        for line in tqdm.tqdm(f, desc='Get Data From Annotation File', total=line_length):
            line_split = line.strip().split(',')
            img_id, img_path, mask_img_path, cls_name = line_split

            # Class Count
            if cls_name not in cls_cnt:
                cls_cnt[cls_name] = 1
            else:
                cls_cnt[cls_name] += 1

            # Class Mapping
            if cls_name not in cls_mapping:
                cls_mapping[cls_name] = len(cls_mapping)

            # Image File Info Dictionary
            if img_id not in img_info:
                img = cv2.imread(img_path)
                rows, cols = img.shape[:2]

                img_info[img_id] = dict(
                    img_id=img_id,
                    img_path=img_path,
                    width=cols,
                    height=rows,
                    masks=list(),
                )

            img_info[img_id]['masks'].append(
                {
                    'class': cls_name,
                    'source': img_id,
                    'class_id': cls_mapping[cls_name],
                    'mask_img_path': mask_img_path,
                }
            )

            # Class Info List
            cls_info_list.append(
                {
                    'source': img_id,
                    'class_name': cls_name,
                    'class_id': cls_mapping[cls_name],
                }
            )

    # Convert img_info to List
    img_info_list = list(img_info.values())
    print('\n==== End of Get Data From Annotation File ====\n')
    return img_info_list, cls_info_list, cls_cnt, cls_mapping
