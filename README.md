# Google_Instance_Segmentation
Kaggle Competition: Open Images 2019 - Instance Segmentation
 - Kaggle Competition: [Open Images 2019 - Instance Segmentation](https://www.kaggle.com/c/open-images-2019-instance-segmentation)
 - Data: [Open Images Dataset V5](https://storage.googleapis.com/openimages/web/index.html)
 
# CoCo Pretrained Model without additional training
  - Preparation
  1. Have python3 installed.
  2. Save Kaggle API key in this path: ``` ~./kaggle/kaggle.json```
  3. Install python libraries:
  ```$ pip install  -r requirements.txt```
  4. Make directories and download data from Open Image Dataset V5 and Kaggle: ``` ./set_env.sh``` 
     
     
  - Reference     
  1. [Mask RCNN](https://github.com/matterport/Mask_RCNN)
  2. Teammate: [Bruce Kim](https://github.com/DevBruce)
  

# Train on Open Dataset Image using maskrcnn-benchmark
  - Preparation
  1. Install maskrcnn_bencmark according to official guide: 
  2. Download the Open Images dataset to the project root directory (or make sim link).
  
  - Create coco format dataset for layer 0 class
  ```linux python create_dataset.py -l 0``` 

  - Train for Layer 0 classes (single GPU)
  ```linux rain.py --config-file config/e2e_mask_rcnn_X_101_32x8d_FPN_1x_1gpu.yaml OUTPUT_DIR "layer0" SOLVER.STEPS "(70000, 100000)" SOLVER.MAX_ITER 120000```
  
  - Test for Layer 0 classes 
  ```linux python test.py -l 0 --weight [TRAINED_WEIGHT_PATH (e.g. layer0/model_0060000.pth)] ```
  
  - Create Submission file for 0 classes
  ```linux python create_submission.py -l 0```
  
  - Create Submission file for 1 classes 
  ```linux python create_dataset.py -l 1  # this overwrite layer 0 dataset. Please move it if needed later
python -m torch.distributed.launch --nproc_per_node=8 train.py OUTPUT_DIR "layer1"
python test.py -l 1 --weight [TRAINED_WEIGHT_PATH (e.g. layer1/model_0060000.pth)]
python create_submission.py -l 1
  ```

  - Integegrate Submission File
  ```linux python integrate_results.py --input1 output_0.csv --input2 output_1.csv```
  
   - Reference     
  1. [Maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md)
  2. If you want to train with multiple GPUS: [yu4u](https://github.com/yu4u/kaggle-open-images-2019-instance-segmentation)
  3. If you want to download yu4u's trained model to go straight into inference phase: [yu4u's trained model](https://www.kaggle.com/ren4yu/openimages2019instancesegmentationmodels)
