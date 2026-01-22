IMAGE_EXTENSION="PNG"
FOOT_GROUND_MODEL_PATH="/opt/detectron2/tools/slowmo2/weights_crop_ResNet_rm.best.hdf5"
INPUT_PATH="/opt/detectron2/tools/slowmo2/Lauren_backsideview.mov"
OUTPUT_DIRECTORY="/opt/detectron2/tools/slowmo2/outputs/"
INPUT_TYPE="video"
INPUT_FPS=30
FACE_BLUR='on'
CAMERA_DISTANCE=10
RESOLUTION="normal"
MAX_FRAME=500
AUTO_RESIZE='off'
PELVIS="on"
FRONT_BACK_PELVIS="on"
SIDE_PELVIS="on"
VIDEO_TYPE="mov"
POSE_DETECTION="off"
POSE_DETECTION="off"
LINE_THICKNESS=2
OUTPUT_OPTION=3
DISTANCE=5
source ~/anaconda3/etc/profile.d/conda.sh
conda activate densepose
python main.py --distance "$DISTANCE" --image-ext "$IMAGE_EXTENSION" --face_blur "$FACE_BLUR" --foot_model_path "$FOOT_GROUND_MODEL_PATH" --input_type "$INPUT_TYPE" --auto_resize "$AUTO_RESIZE" --auto_fps "$AUTO_FPS" --input_fps "$INPUT_FPS" --cam-distance "$CAMERA_DISTANCE" --raw_video "$RESOLUTION" --pelvis "$PELVIS" --video_type "$VIDEO_TYPE" --pose_detection "$POSE_DETECTION" --line_thicknes "$LINE_THICKNESS" --max-frame "$MAX_FRAME" --output-dir "$OUTPUT_DIRECTORY" --video_path "$INPUT_PATH" --output-option "$OUTPUT_OPTION" --side_pelvis "$SIDE_PELVIS" --front_back_pelvis "$FRONT_BACK_PELVIS"
