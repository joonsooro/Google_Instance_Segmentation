#!/bin/bash
echo -e "Before start set environment, you have to set your kaggle API at ~/.kaggle\n"
echo -e "\n=== Start Create Basic Directory ===\n"
mkdir ./data
mkdir ./data/meta_data
mkdir ./data/train_img
mkdir ./data/train_mask_img
mkdir ./data/val_img
mkdir ./data/val_mask_img
mkdir ./data/test_img
mkdir ./data/downloaded_raw
mkdir ./pretained_model
echo -e "\n*** Create Basic Directory Completed ***\n"
echo -e "\n=== Start Download Metadata ===\n"
wget https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-classes-description-segmentable.csv -O ./data/meta_data/challenge-2019-classes-description-segmentable.csv
wget https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-segmentation-masks.csv -O ./data/meta_data/challenge-2019-train-segmentation-masks.csv
wget https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-segmentation-masks.csv -O ./data/meta_data/challenge-2019-validation-segmentation-masks.csv
echo -e "\n*** Download Metadata Completed ***\n"
echo -e "\n=== Start Download COCO Weight File ===\n"
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 -O ./pretained_model/mask_rcnn_coco.h5
echo -e "\n*** Download COCO Weight File Completed ***\n"
echo -e "\n=== Start Download Image (Train & Validation) Data ===\n"
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_0.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_1.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_2.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_3.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_4.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_5.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_6.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_7.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_8.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_9.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_a.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_b.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_c.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_d.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_e.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_f.tar.gz ./data/downloaded_raw
aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz ./data/downloaded_raw
echo -e "\n*** Download Image Data (Train & Validation) Completed ***\n"
echo -e "\n=== Start Download Train Mask Image Data ===\n"
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-0.zip -O ./data/downloaded_raw/challenge-2019-train-masks-0.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-1.zip -O ./data/downloaded_raw/challenge-2019-train-masks-1.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-2.zip -O ./data/downloaded_raw/challenge-2019-train-masks-2.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-3.zip -O ./data/downloaded_raw/challenge-2019-train-masks-3.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-4.zip -O ./data/downloaded_raw/challenge-2019-train-masks-4.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-5.zip -O ./data/downloaded_raw/challenge-2019-train-masks-5.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-6.zip -O ./data/downloaded_raw/challenge-2019-train-masks-6.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-7.zip -O ./data/downloaded_raw/challenge-2019-train-masks-7.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-8.zip -O ./data/downloaded_raw/challenge-2019-train-masks-8.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-9.zip -O ./data/downloaded_raw/challenge-2019-train-masks-9.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-a.zip -O ./data/downloaded_raw/challenge-2019-train-masks-a.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-b.zip -O ./data/downloaded_raw/challenge-2019-train-masks-b.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-c.zip -O ./data/downloaded_raw/challenge-2019-train-masks-c.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-d.zip -O ./data/downloaded_raw/challenge-2019-train-masks-d.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-e.zip -O ./data/downloaded_raw/challenge-2019-train-masks-e.zip
wget https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-f.zip -O ./data/downloaded_raw/challenge-2019-train-masks-f.zip
echo -e "\n*** Download Train Mask Image Data Completed ***\n"
echo -e "\n=== Start Download Validation Mask Image Data ===\n"
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-0.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-0.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-1.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-1.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-2.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-2.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-3.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-3.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-4.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-4.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-5.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-5.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-6.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-6.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-7.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-7.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-8.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-8.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-9.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-9.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-a.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-a.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-b.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-b.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-c.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-c.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-d.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-d.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-e.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-e.zip
wget https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-f.zip -O ./data/downloaded_raw/challenge-2019-validation-masks-f.zip
echo -e "\n*** Download Validation Mask Image Data Completed ***\n"
echo -e "\n=== Download Competition Data (Test Image & Empty Submission) ===\n"
kaggle competitions download -c open-images-2019-instance-segmentation -p ./data/downloaded_raw
echo -e "\n*** Download Competition Data (Test Image & Empty Submission) Completed ***\n"
echo -e "\n=== Start Unzip All Image Data ===\n"
tar -xzf ./data/downloaded_raw/train_0.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_1.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_2.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_3.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_4.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_5.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_6.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_7.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_8.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_9.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_a.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_b.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_c.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_d.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_e.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/train_f.tar.gz -C ./data/train_img --strip-components=1
tar -xzf ./data/downloaded_raw/validation.tar.gz -C ./data/val_img --strip-components=1
unzip ./data/downloaded_raw/test.zip -d ./data/test_img
mv ./data/downloaded_raw/sample_empty_submission.csv ./data/meta_data/
echo -e "\n*** Unzip All Image Data Completed ***\n"
echo -e "\n=== Start Unzip All Mask Image Data ===\n"
unzip ./data/downloaded_raw/challenge-2019-train-masks-0.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-1.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-2.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-3.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-4.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-5.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-6.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-7.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-8.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-9.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-a.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-b.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-c.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-d.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-e.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-train-masks-f.zip -d ./data/train_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-0.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-1.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-2.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-3.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-4.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-5.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-6.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-7.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-8.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-9.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-a.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-b.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-c.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-d.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-e.zip -d ./data/val_mask_img/
unzip ./data/downloaded_raw/challenge-2019-validation-masks-f.zip -d ./data/val_mask_img/
echo -e "\n*** Unzip All Mask Image Data Completed ***\n"
echo -e "\n=== Delete Downloaded Raw Files ===\n"
rm -rf ./data/downloaded_raw
echo -e "\n*** Delete Downloaded Raw Files Completed ***\n"
echo -e "\n >>> *** All Process Completed *** <<<\n"
echo -e "\t==> Now you can create annotation file with [./create_annotation.py] and train the model with data\n"
