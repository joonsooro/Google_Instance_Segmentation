import os
import pickle
import tqdm
import skimage.io
import pandas as pd
from sub_func.encode_mask import encode_binary_mask
from config import TEST_IMG_DIR, MODEL_DIR


with open(os.path.join(MODEL_DIR, 'train_class_mapping.pickle'), 'rb') as f:
    TRAIN_CLASS_MAPPING = pickle.load(f)
TRAIN_CLASS_MAPPING_REV = {val: key for key, val in TRAIN_CLASS_MAPPING.items()}


def get_submission(model, empty_submission_path, class_description_path):
    df_empty_submission = pd.read_csv(empty_submission_path)
    df_class_description = pd.read_csv(class_description_path, names=['encoded_label', 'label'])

    ImageID_list = list()
    ImageHeight_list = list()
    ImageWidth_list = list()
    PredictionString_list = list()

    for num, row in tqdm.tqdm(
            df_empty_submission.iterrows(),
            desc='Get submission process',
            total=df_empty_submission.shape[0],
    ):
        filename = row['ImageID'] + '.jpg'
        image = skimage.io.imread(os.path.join(TEST_IMG_DIR, filename))
        results = model.detect([image])
        r = results[0]

        PredictionString = ''
        for i in range(len(r['class_ids'])):
            class_id = r['class_ids'][i]
            mask = r['masks'][:, :, i]
            confidence = r['scores'][i]
            encoded_mask = encode_binary_mask(mask)

            label_name = TRAIN_CLASS_MAPPING_REV[class_id]
            if df_class_description[df_class_description['label'] == label_name].shape[0] == 0:
                # no match label
                continue

            encoded_label = df_class_description[df_class_description['label'] == label_name]['encoded_label'].item()
            PredictionString += encoded_label + ' ' + str(confidence) + ' ' + encoded_mask.decode() + ' '

        ImageID_list.append(row['ImageID'])
        ImageHeight_list.append(image.shape[0])
        ImageWidth_list.append(image.shape[1])
        PredictionString_list.append(PredictionString)

    submission = pd.DataFrame({
        'ImageID': ImageID_list,
        'ImageWidth': ImageWidth_list,
        'ImageHeight': ImageHeight_list,
        'PredictionString': PredictionString_list,
    })
    print('\n === Get Submission Completed ===\n [submission.csv] will be created soon.')
    return submission
