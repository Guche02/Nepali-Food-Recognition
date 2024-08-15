import os
import sys
import json
import numpy as np
import skimage.draw
import cv2
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from keras.callbacks import CSVLogger
import csv
import imgaug.augmenters as iaa

# Root directory of the project
ROOT_DIR = "D:\Youtube_MaskRCNN"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained COCO weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class CustomConfig(Config):
    """Configuration for training on the custom dataset."""
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 4  # Background + egg, rice, lentils, spinach
    BATCH_SIZE = 4
    STEPS_PER_EPOCH = 3
    DETECTION_MIN_CONFIDENCE = 0.8

class CustomDataset(utils.Dataset):
    """Custom dataset class."""
    def load_custom(self, dataset_dir, subset):
        """Load dataset."""
        self.add_class("object", 1, "egg")
        self.add_class("object", 2, "rice")
        self.add_class("object", 3, "lentils")
        self.add_class("object", 4, "spinach")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(open(os.path.join(dataset_dir, 'via_region_data.json')))
        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
            image_id = "{}_{}".format(subset, a['filename'])
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['name'] for s in a['regions']]
            name_dict = {"egg": 1, "rice": 2, "lentils": 3, "spinach":4}
            num_ids = [name_dict[a] for a in objects]

            self.add_image(
                "object",
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    dataset_train = CustomDataset()
    dataset_train.load_custom("D:\Youtube_MaskRCNN\dataset", "train")
    dataset_train.prepare()

    dataset_val = CustomDataset()
    dataset_val.load_custom("D:\Youtube_MaskRCNN\dataset", "val")
    dataset_val.prepare()

    csv_logger = CSVLogger(os.path.join(DEFAULT_LOGS_DIR, 'history.csv'), append=True)

    custom_callbacks = [csv_logger]

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads',
                custom_callbacks=custom_callbacks
                )

# Configuration and model initialization
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

# Load COCO weights
weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"
])

# Start training
train(model)
