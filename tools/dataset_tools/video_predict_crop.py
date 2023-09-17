import argparse

import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
import os
import torch
import uuid  # Import the uuid library
from tqdm import tqdm
import numpy as np

def cut_video(video_path, save_image, model):   
    
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    video_reader = mmcv.VideoReader(video_path)
    total_frames = video_reader.frame_cnt
    frame_count =  0
    vehicle_count = 0
    desired_fps = 5
    fps = int(video_reader.fps)
    frame_interval = int(fps / desired_fps)
    frame_save = 0
    for frame in track_iter_progress(video_reader):
        frame_count += 1
        # Only process every frame_interval frames
        if frame_count % frame_interval == 0:
            frame_save += 1
            result = inference_detector(model, frame, test_pipeline=test_pipeline)
            # Get results labels
            pred_classes = result.pred_instances.labels # Tensor of all predict labels index
            # take index where there is class = 2, 5, 6, 7, 8
            car = torch.where(pred_classes == 2)[0] # 2 is car
            bus = torch.where(pred_classes == 5)[0] # 5 is bus
            train = torch.where(pred_classes == 6)[0] # 6 is truck
            truck = torch.where(pred_classes == 7)[0] # 7 is train
            boat = torch.where(pred_classes == 8)[0] # 8 is boat
            vehicle = torch.cat((car, bus, train, truck, boat), 0) # Tensor of all vehicle index
            bagac = torch.where(pred_classes == 13)[0] # Tensor of all bagac index

            # Keep index if its score is greater than 0.3
            pred_scores = result.pred_instances.scores # Tensor of predict scores
            thresh_1_index = torch.where(pred_scores > 0.3)[0] # Tensor of index where score > 0.3
            # Keep only vehicle index where score > 0.3
            vehicle_keep = vehicle[torch.isin(vehicle, thresh_1_index)]
            thresh_2_index = torch.where(pred_scores > 0.1)[0] # Tensor of index where score > 0.1
            # Keep only bagac index where score > 0.1
            bagac_keep = bagac[torch.isin(bagac, thresh_2_index)]

            # Get bounding boxes
            pred_boxes = result.pred_instances.bboxes
            vehicle_boxes = pred_boxes[vehicle_keep]
            bagac_boxes = pred_boxes[bagac_keep]
            combined_boxes = torch.cat((vehicle_boxes, bagac_boxes), 0)
            print(' Found {} vehicles and {} bagac  '.format(len(vehicle_boxes), len(bagac_boxes)))
            vehicle_count += len(combined_boxes)
            # Crop image
            image = frame
            h, w, _ = image.shape
            for idx, bbox in enumerate(combined_boxes):
                # Scale bounding box 0.1 H, 0.05 W , type XYXY
                box_height = bbox[3] - bbox[1]
                box_width = bbox[2] - bbox[0]
                bbox[0] = max(int(bbox[0] - box_width * 0.05), 0)
                bbox[1] = max(int(bbox[1] - box_height * 0.1), 0)
                bbox[2] = min(int(bbox[2] + box_width * 0.05), w)
                bbox[3] = min(int(bbox[3] + box_height * 0.1), h)
                # Crop the image using the bounding box
                cropped_image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                # Generate cropped image filename  
                unique_key = uuid.uuid4().hex
                cropped_image_filename = f'frame_{frame_save}_{idx}_{unique_key}.jpg'
                # Save the cropped image
                cropped_image_path = os.path.join(save_image, cropped_image_filename)
                cv2.imwrite(cropped_image_path, cropped_image)
    print('Found {} vehicles in total'.format(vehicle_count))
def arg_parse():
    parser = argparse.ArgumentParser(description='MMDetection video inference')
    parser.add_argument('--video_dir', help='video file')
    parser.add_argument('--save', help='save image')
    args = parser.parse_args()
    return args
def main():
    args = arg_parse()
    config = 'configs/mmdetection/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
    checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth'
    video_dir = args.video_dir
    save_image = args.save
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    model = init_detector(config, checkpoint, device='cuda:0')
    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    count = 0
    for video in os.listdir(video_dir):
        count +=1
        if count > 30:
            break
        print('Processing video {} of {}'.format(count, 30))
        video_path = os.path.join(video_dir, video)
        cut_video(video_path, save_image, model)
if __name__ == '__main__':
    main()