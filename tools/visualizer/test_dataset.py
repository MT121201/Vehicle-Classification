import os
import mmcv
import click
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmpretrain.utils import register_all_modules
# import cvut.draw as cvutdraw
register_all_modules()
from mmpretrain.registry import DATASETS

@click.command()
@click.option("--config_file", default="configs/tnivc/mobilenet-v3-small_8xb128_vc.py", help="your config file")
@click.option("--save_img_dir", default='./cache/debugdata', help="save image dir")
def main(config_file, save_img_dir):
    cfg = Config.fromfile(config_file)

    # build dataset
    dataset_cfg = cfg['train_dataloader']['dataset']
    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)

    num_draw = min(100, len(dataset))

    print('len dataset: ', len(dataset))

    # take mean and std
    # mean = np.array(cfg.img_norm_cfg['mean'])
    # std = np.array(cfg.img_norm_cfg['std'])

    # draw all
    try:
        os.system(f'rm -rf {save_img_dir}')
    except:
        pass
    os.makedirs(save_img_dir, exist_ok=True)


    print('save images to: ', save_img_dir)

    for idx in tqdm(range(0, num_draw, 1)):

        sample = dataset[idx]

        # Read images
        img_name = os.path.basename(sample['data_samples'].img_path)
        img = sample['inputs'].data.cpu().numpy().transpose(1, 2, 0)

        # Convert back
        # img = np.clip(img * std + mean, 0, 255).astype('uint8')

        # gt_bboxes = sample['data_samples']..bboxes.tensor.numpy().astype(np.int16)
        # gt_labels = sample['data_samples'].gt_instances.labels.numpy().astype(np.int16)

        # # Draw bboxes
        # img = cvutdraw.draw_bboxes(img, gt_bboxes, color=(255, 0, 0))

        # # Draw polygon
        # if 'gt_polygons' in sample:
        #     polygons = sample['gt_polygons'].data.cpu().numpy()
        #     img = cvutdraw.draw_polygons(img, polygons, thickness=3)

        save_file = os.path.join(save_img_dir, img_name)
        mmcv.imwrite(img, save_file)
        print(f'image is saved at: {save_file}')

if __name__ == '__main__':
    main()
