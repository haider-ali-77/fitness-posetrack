from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import cv2

if __name__ == "__main__":
    print("Main Function")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "DensePose/configs/densepose_rcnn_R_101_FPN_s1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "DensePose/configs/densepose_rcnn_R_101_FPN_s1x.yaml")
    model = DefaultPredictor(cfg)
    
    im = cv2.imread("46.PNG")
    model_output = model(im)
    print(model_output)