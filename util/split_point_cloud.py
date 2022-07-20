import os
import glob
import numpy as np
from pyntcloud import PyntCloud
from collections import defaultdict
from tqdm import tqdm
from itertools import cycle
from torch.multiprocessing import Pool

def partition(filename, args):
    # partition.
    point_cloud = PyntCloud.from_file(filename)
    pc_xyz = point_cloud.points.values[:, :3]

    # partition point cloud to cubes.
    cubes = defaultdict(list)
    for point in pc_xyz:
        cube_index = tuple((point//args.cube_size).astype("int"))
        local_point = point % args.cube_size # np.array
        cubes[cube_index].append(local_point)
    list_item = list(cubes.keys())
    for item in list_item:
        if len(cubes[item]) < args.min_num:
            del cubes[item]
    return cubes

def write_cubes(cubes, args, name):
    items = os.path.basename(name).replace('.ply', '').split('_')
    if 'enc' in name:
        category = items[-2]
        frame = items[-3].zfill(8)
        squence = items[1]
    else:
        category = 'gt'
        squence = items[0]
        frame = items[-1].zfill(8)
    result_dir = '{}/{}/{}'.format(category, squence, frame)
    result_dir = os.path.join(args.save_dir, result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for item in cubes:
        name = [str(item).zfill(3) for item in item]
        name = os.path.join(result_dir, '_'.join(name)+'.txt')
        array = np.vstack(cubes[item]).astype(np.int32)
        np.savetxt(name, array, '%d')

def run(data):
    filename, args = data
    cubes = partition(filename, args)
    write_cubes(cubes, args, filename)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_dir", type=str, default="./example")
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--cube_size", type=int, default=64)
    parser.add_argument("--min_num", type=int, default=3)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()

    point_cloud_dirs = sorted(glob.glob(os.path.join(args.source_dir, '*.ply')))

    pool = Pool(args.workers)
    args_list = cycle([args])
    for data in tqdm(pool.imap_unordered(run, zip(point_cloud_dirs, args_list))):
        None    
