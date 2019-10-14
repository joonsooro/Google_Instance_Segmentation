from config import *
import mrcnn.model as modellib

# Data Prepare
TRAIN_DATASET.prepare_modified(class_map=TRAIN_CLASS_MAPPING, mode='Train')
VAL_DATASET.prepare_modified(class_map=TRAIN_CLASS_MAPPING, mode='Validation')

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=CONFIG,
                          model_dir=MODEL_DIR)

if TRAIN_INIT_WITH == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif TRAIN_INIT_WITH == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif TRAIN_INIT_WITH == 'direct_set':
    model.load_weights(DIRECT_MODEL_PATH, by_name=True)

# Train
model.train(
    train_dataset=TRAIN_DATASET,
    val_dataset=VAL_DATASET,
    learning_rate=CONFIG.LEARNING_RATE,
    epochs=EPOCHS,
    layers='all',  # Fine-tune all layers. ('all', 'heads')
    augmentation=None,
    custom_callbacks=None,
    no_augmentation_sources=None,
)
