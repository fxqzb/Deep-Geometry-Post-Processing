from genericpath import exists
import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch

from config import Config
from util.distributed import init_dist
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from data.demo_point_cloud_dataset import DemoPointCloudDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/fashion_512.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result', help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default='eval_result')
    parser.add_argument('--threshold', type=str, default='fixed')
    parser.add_argument('--threshold_value', type=float, default=0.98)
    parser.add_argument('--test_list', type=str, default='./test.txt')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get training options
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)

    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = torch.cuda.current_device()
    else:
        opt.distributed = False
        opt.device = torch.cuda.current_device()


    opt.logdir = os.path.join(opt.checkpoints_dir, opt.name) 
    print(opt.logdir)   
    # create a model
    net_G, net_G_ema, opt_G, sch_G \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_G_ema, \
                          opt_G, sch_G, \
                          None)

    current_epoch, current_iteration = trainer.load_checkpoint(
        opt, args.which_iter)                          
    output_dir = os.path.join(
        args.output_dir, 
        'epoch_{:05}_iteration_{:09}'.format(current_epoch, current_iteration)
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    net_G = net_G_ema.eval()
    data_loader = DemoPointCloudDataset(opt.data)
    test_dict = {}
    with open(args.test_list, 'r') as f:
        lines = f.readlines()
        lines = [item.strip().split(',') for item in lines]
        test_dict['input'] = [item[0] for item in lines]
        test_dict['gt'] = [item[1] for item in lines]
    num_files = len(test_dict['input'])
    with torch.no_grad():
        for index in tqdm(range(num_files)):
            start_time = time.time()
            cubes_input, cubes_gt = data_loader.load_item(test_dict['input'][index], test_dict['gt'][index])
            predict = []
            input, global_id_list = data_loader.convert_to_onehot(cubes_input)
            num_cubes = input.shape[0]
            predict = []
            for i in range(0, num_cubes, data_loader.batch_size):
                input_batch = input[i:i+data_loader.batch_size, None, ...].to(opt.device)
                predict_batch = net_G(input_batch)
                predict.append(predict_batch.detach().cpu())
                # predict.append(predict_batch[2].detach().cpu())
            predict = torch.cat(predict, 0)

            final_cubes = {}
            for i, global_id in enumerate(global_id_list):
                item_predict = predict[i, 0]
                if args.threshold == 'fixed':
                    item_predict = (item_predict > args.threshold_value).numpy().astype(np.int32)
                elif args.threshold == 'adaptive':
                    num_point = len(cubes_gt[global_id])
                    if num_point != 0:
                        threshold_value = item_predict.view(-1).sort()[0][-num_point]
                    else:
                        threshold_value = 1
                    item_predict = (item_predict > threshold_value).numpy().astype(np.int32)
                else:
                    assert False
                points = np.array(np.where(item_predict>0)).transpose((1, 0))                  
                final_cubes[global_id] = points
            end_time = time.time()
            print('process needs: %.2f s', end_time-start_time)
            write_name = os.path.join(output_dir, os.path.basename(test_dict['input'][index]))
            data_loader.write_to_ply(final_cubes, write_name)

