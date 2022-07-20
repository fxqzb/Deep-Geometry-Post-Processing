import os
import collections
import numpy as np
from pyntcloud import PyntCloud

import torch


class DemoPointCloudDataset(object):
    def __init__(self, args):
        super().__init__()
        self.input_cube_size = args.cube_size
        self.cube_size = 64
        self.strid = 64
        self.batch_size = args.val.batch_size

    def partition(self, filename):
        # partition.
        point_cloud = PyntCloud.from_file(filename)
        pc_xyz = point_cloud.points.values[:, :3]

        # partition point cloud to cubes.
        cubes = collections.defaultdict(list)
        for point in pc_xyz:
            cube_index = tuple((point//self.cube_size).astype("int"))
            local_point = point % self.cube_size # np.array
            cubes[cube_index].append(local_point)
        return cubes

    def load_item(self, filename_input, filename_gt):
        cubes_input = self.partition(filename_input)
        cubes_gt = self.partition(filename_gt)

        return cubes_input, cubes_gt

    def write_to_ply(self, cubes, filename, cube_size=None):
        """Combine & save points."""
        if os.path.isfile(filename):
            os.remove(filename)
            print('removing the old file {}'.format(filename))
        cube_size = cube_size if cube_size is not None else self.cube_size
        # combine points.
        point_cloud = []
        for index, point in cubes.items():
            point = point + np.array(index) * cube_size
            point_cloud.append(point)
        
        self.write_ply_data(filename, np.vstack(point_cloud))

      
    def convert_to_onehot(self, cubes):
        inputs_list, global_id_list = [] , []
        for item in cubes:
            points = np.vstack(cubes[item])
            points = (points).astype(np.int32)
            inputs = np.zeros((self.cube_size, self.cube_size, self.cube_size)) 
            inputs[points[:,0], points[:,1], points[:,2]] = 1 
            inputs_list.append(torch.tensor(inputs).type(torch.float32)[None, ])
            global_id_list.append(item)
        inputs = torch.cat(inputs_list, 0)
        return inputs, global_id_list

    def write_ply_data(self, filename, points):
        f = open(filename, 'a+')
        f.writelines(['ply\n','format ascii 1.0\n'])
        f.write('element vertex '+str(points.shape[0])+'\n')
        f.writelines(['property float x\n','property float y\n','property float z\n'])
        f.write('end_header\n')
        for _, point in enumerate(points):
            f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), '\n'])
        f.close() 