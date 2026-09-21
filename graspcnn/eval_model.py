import argparse
import logging
import torch

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--model', type=str, help='Path to saved network to evaluate')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')
    parser.add_argument('--n-grasps', type=int, default=2, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # Load the model
    model = torch.load(args.model)
    device = torch.device("cuda:0")
    # Load the dataset
    logging.info('Loading Dataset...')
    Dataset = get_dataset("cornell")
    test_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=0.0,
                           random_rotate=args.augment, random_zoom=args.augment,
                           include_depth=args.use_depth, include_rgb=args.use_rgb)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info('Done')
    results = {'correct': 0, 'failed': 0}

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing Scene: {}/{}'.format(idx+1, len(test_data)))
            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = model.compute_loss(xc, yc)
            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])
            if args.iou_eval:
                success = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom), no_grasps=args.n_grasps, grasp_width=width_img)
                if success:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            if args.vis:
                evaluation.plot_output(test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                                       test_data.dataset.get_depth(didx, rot, zoom), q_img,
                                       ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img, id=idx)

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))