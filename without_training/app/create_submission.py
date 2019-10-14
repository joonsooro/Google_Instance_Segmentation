from config import OpenImgConfig, MODEL_DIR, DIRECT_MODEL_PATH, EMPTY_SUBMISSION_PATH, CLASS_DESCRIPTION_PATH, SUBMISSION_PATH
from sub_func.get_submission import get_submission
import mrcnn.model as modellib


class InferenceConfig(OpenImgConfig):
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()
# Recreate the model in inference mode
model = modellib.MaskRCNN(
    mode="inference",
    config=inference_config,
    model_dir=MODEL_DIR,
)

# Load trained weights
print("Loading weights from ", DIRECT_MODEL_PATH)
model.load_weights(DIRECT_MODEL_PATH, by_name=True)

# Create Submission DataFrame
submission = get_submission(
    model=model,
    empty_submission_path=EMPTY_SUBMISSION_PATH,
    class_description_path=CLASS_DESCRIPTION_PATH,
)

# Save Submission File
submission.to_csv(SUBMISSION_PATH, index=False)
print('\n === Create [submission.csv] Completed ===')
