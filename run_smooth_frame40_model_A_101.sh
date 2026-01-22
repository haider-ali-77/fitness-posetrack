#!/bin/bash
# export PYTHONPATH=/usr/local/lib/:/usr/local/include/caffe2:/usr/local/lib/python2.7/dist-packages/caffe2:$PYTHONPATH

IMAGE_EXTENSION="PNG"
#INPUT_PATH="/home/stevegrosserode/input_still_frame/frames/"
#Video: PATH="/home/stevegrosserode/input_multiple_videos/"
# INPUT_PATH="/home/stevegrosserode/input_multiple_videos/"
# trisha_othersinframe_backview_normalspeed_IMG_2497.mov
# trisha_front_slomo_IMG_2498
# trisha_front_slomo_IMG_2498.mov
# trisha_left_20ft_slomo_IMG_2496.mov
# trisha_right_20ft_slomo_IMG_2495.mov
# test_backview.mp4
# lauren_backview
# /opt/videos/input/new_video/treadmill_5ft_away.mov
# INPUT_PATH="/home/stevegrosserode/input_multiple_videos/"
# OUTPUT_DIRECTORY="/home/stevegrosserode/output_multiple_videos_smooth_frame_version_4.t
#INPUT_PATH="/opt/detectron2/tools/101/data/videos/Backview_Indoor_#1_images_indoor.mov"
INPUT_PATH="/opt/detectron2/tools/slowmo2/backview_PROPT_daylight_good_model.mov"
OUTPUT_DIRECTORY="/opt/detectron2/tools/slowmo2/outputs/"
FOOT_GROUND_MODEL_PATH="/opt/detectron2/tools/slowmo2/weights_crop_ResNet_rm.best.hdf5"


# INPUT_PATH="/home/stevegrosserode/input_multiple_videos/"
#Video: PATH="/home/stevegrosserode/input_multiple_videos/"
#Video: OUTPUT_DIRECTORY="/home/stevegrosserode/output_multiple_videos_smooth_frame/"
#Image: INPUT_PATH="/home/stevegrosserode/input_still_frames/frames/"
#Image: OUTPUT_DIRECTORY="/home/stevegrosserode/output_still_frame/"
#INPUT_PATH="/home/stevegrosserode/input_multiple_videos/"
#OUTPUT_DIRECTORY="/home/stevegrosserode/output_multiple_videos_smooth_frame_version_3.8/"
# OUTPUT_DIRECTORY="/home/stevegrosserode/output_multiple_videos_smooth_frame_version_4.0/"
# OUTPUT_DIRECTORY="/home/stevegrosserode/output_multiple_videos_smooth_frame_version_4.2/"
# OUTPUT_DIRECTORY="/home/stevegrosserode/output_multiple_videos_smooth_frame_version_4.0/"

##INPUT_TYPE="video" or "image"
#INPUT_TYPE="image"
INPUT_TYPE="video"
INPUT_FPS=60
# Auto select fps or use input frames on or off
#AUTO_FPS="on"
CAMERA_DISTANCE=10
####RESOLUTION="normal" or "raw"
RESOLUTION="normal"
MAX_FRAME=50
#PELVIS="on" or "off"
# PELVIS="off"
PELVIS="off"
###MULTIPLE_VIDEO="on"
VIDEO_TYPE="mov"
POSE_DETECTION="off"
LINE_THICKNESS=2
### 1=old, 2=new, 3=both
OUTPUT_OPTION=3
# tools/101/smooth_frame_keypoint_v4_1_101.py
# person distance from camera
DISTANCE=5
python main.py --distance "$DISTANCE" --image-ext "$IMAGE_EXTENSION" --foot_model_path "$FOOT_GROUND_MODEL_PATH" --input_type "$INPUT_TYPE" --auto_fps "$AUTO_FPS" --input_fps "$INPUT_FPS" --cam-distance "$CAMERA_DISTANCE" --raw_video "$RESOLUTION" --pelvis "$PELVIS" --video_type "$VIDEO_TYPE" --pose_detection "$POSE_DETECTION" --line_thicknes "$LINE_THICKNESS" --max-frame "$MAX_FRAME" --output-dir "$OUTPUT_DIRECTORY" --video_path "$INPUT_PATH" --output-option "$OUTPUT_OPTION"


#python /opt/detectron2/tools/101/smooth_frame_keypoint_v4_4_101.py --distance "$DISTANCE" --output-dir "$OUTPUT_DIRECTORY" --output-option "$OUTPUT_OPTION" --image-ext "$IMAGE_EXTENSION" --video_path "$INPUT_PATH" --input_type "$INPUT_TYPE" --input_fps $INPUT_FPS --cam-distance $CAMERA_DISTANCE --raw_video "$RESOLUTION" --max-frame $MAX_FRAME --pelvis "$PELVIS" --video_type "$VIDEO_TYPE" --pose_detection "$POSE_DETECTION" --line_thicknes $LINE_THICKNESS

