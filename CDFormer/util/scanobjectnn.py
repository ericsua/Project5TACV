import os
import sys
import h5py
import pickle
import numpy as np
sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset
from util.data_util import data_prepare_scanobjnn as data_prepare


import matplotlib.pyplot as plt
from util.shapenet10 import ShapeNet10

class ScanObjectNNHardest(Dataset):
    """The hardest variant of ScanObjectNN.
    The data we use is: `training_objectdataset_augmentedrot_scale75.h5`[1],
    where there are 2048 points in training and testing.
    The number of training samples is: 11416, and the number of testing samples is 2882.
    Args:
    """
    classes = [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    ]
    classes_common_mn40 = [
    'cabinet', 'chair', 'desk', 'display', 'door',
    'shelf', 'table', 'bed', 'sink', 'sofa', 'toilet']
    
    classes_common_pointda = [
        'bed', 'shelf', 'cabinet', 'chair', 
        'display', 'sofa', 'table'
    ]
    
    num_classes = 15
    gravity_dim = 1

    def __init__(self, data_dir, split,
                 num_points=2048,
                 uniform_sample=True,
                 transform=None,
                 **kwargs):
        super().__init__()
        self.partition = split
        self.transform = transform
        self.num_points = num_points
        slit_name = 'training' if split == 'train' else 'test'
        h5_name = os.path.join(
            data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75.h5')

        if not os.path.isfile(h5_name):
            raise FileExistsError(
                f'{h5_name} does not exist, please download dataset at first')
        with h5py.File(h5_name, 'r') as f:
            self.points = np.array(f['data']).astype(np.float32)
            self.labels = np.array(f['label']).astype(int)
            
        fig = plt.figure(figsize=(16, 12), dpi=300)
        ax = fig.add_subplot(projection='3d')

        item40 = self.points[12]
        ax.scatter(
        item40[:, 0], item40[:, 1], item40[:, 2],
        s=8.0, alpha=0.5, c='red', cmap='Paired')
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])
        # Show axes
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        
        plt.savefig("plot/bathtub_sonn.png")
        
        # Calculate the norm for each point in each point cloud
        norms = np.linalg.norm(self.points, axis=2)

        # Find the maximum norm for each point cloud
        max_norms = np.max(norms, axis=1)

        # Calculate the mean of the maximum norms
        mean_max_norm = np.mean(max_norms)
        print("Average norms: ", mean_max_norm)
        # if slit_name == 'test' and uniform_sample:
        #     precomputed_path = os.path.join(
        #         data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
        #     if not os.path.exists(precomputed_path):
        #         raise FileExistsError(
        #             f'{precomputed_path} does not exist, please compute at first')
        #     else:
        #         with open(precomputed_path, 'rb') as f:
        #             self.points = pickle.load(f)
        #             print(f"{precomputed_path} load successfully")
        print(f'Successfully load ScanObjectNN {split} '
              f'size: {self.points.shape}, num_classes: {self.labels.max()+1}')

    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, idx):
        coord = self.points[idx][:self.num_points]
        #coord = self.points[idx]
        label = self.labels[idx]
        
        #print(f'idx: {idx}, label: {label}, shape coord: {coord.shape}, num_points: {self.num_points}')

        if self.partition == 'train':
            np.random.shuffle(coord)

        if self.transform is not None:
            coord, _ = self.transform(coord, coord.copy())  # (coord, feat)

        height = coord[:, self.gravity_dim:self.gravity_dim+1]
        height = height - height.min()

        feat = np.concatenate((coord, height), axis=1)

        feat = torch.tensor(coord, dtype=torch.float)
        #feat = torch.tensor(feat, dtype=torch.float)
        coord = torch.tensor(coord, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return coord, feat, label

    def __len__(self):
        return self.points.shape[0]

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """

def index_nested(nested_list, target):
    for i, element in enumerate(nested_list):
        if isinstance(element, list):
            if target in element:
                return i
        else:
            if element == target:
                return i
    return -1


class ScanObjectNNHardest_mn40(ScanObjectNNHardest):
    def __init__(self, data_dir, split, num_points=2048, uniform_sample=True, transform=None, **kwargs):
        super().__init__(data_dir, split, num_points, uniform_sample, transform, **kwargs)
        
        filtered_points = list()
        filtered_labels = list()
        for pts, lbl in zip(self.points, self.labels):
            if self.classes[lbl] in self.classes_common_mn40: # if class is in common with mn40, no nested needed since we only have one level for sonn
            #if index_nested(self.classes_common_mn40, self.classes[lbl]) != -1 : # if class is in common with mn40
                filtered_points.append(pts)
                # get corresponding label index in mn40 common classes
                #idx = index_nested(ModelNet40Ply2048.classes_common_sonn, self.classes[lbl])
                idx = self.classes_common_mn40.index(self.classes[lbl])
                if idx == -1:
                    raise ValueError(f'1 Class {self.classes[lbl]} not found in ModelNet40Ply2048.classes_common_sonn')
                filtered_labels.append(idx) # not the final label of mn40, but the index of common classes (could be a list), in test, we need to convert it to the final label of mn40
        self.points = np.array(filtered_points)
        self.labels = np.array(filtered_labels)
        
    def __getitem__(self, idx):
        #coord = self.points[idx][:self.num_points]
        #print(self.points[idx][:self.num_points].shape)
        coord = self.points[idx][:self.num_points]#[:, [0,2,1]]
        #coord = normalize_point_cloud(coord)
        #coord = self.points[idx]
        label = self.labels[idx]
        
        #print(f'idx: {idx}, label: {label}, shape coord: {coord.shape}, num_points: {self.num_points}')

        if self.partition == 'train':
            np.random.shuffle(coord)

        if self.transform is not None:
            coord, _ = self.transform(coord, coord.copy())  # (coord, feat)

        height = coord[:, self.gravity_dim:self.gravity_dim+1]
        height = height - height.min()

        feat = np.concatenate((coord, height), axis=1)

        feat = torch.tensor(coord, dtype=torch.float)
        #feat = torch.tensor(feat, dtype=torch.float)
        coord = torch.tensor(coord, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return coord, feat, label
            
        
    
if __name__ == '__main__':
    # data_dir = 'dataset/scanobjectnn/h5_files/main_split'
    # split = 'train'
    # num_points = 1024 if split == 'train' else 1024
    # dataset = ScanObjectNNHardest(data_dir, split, num_points=num_points)
    # maxs, mins, means = [], [], []
    # for i in range(len(dataset)):
    #     coord, feat, label = dataset.__getitem__(i)
    #     print(f'{i:05} shape:{coord.size()}{feat.shape} label:{label}')
    #     assert coord.shape[0] == num_points
    #     maxs.append(coord.max().item())
    #     mins.append(coord.min().item())
    #     means.append(coord.mean().item())
    # print(np.max(maxs), np.min(mins), np.mean(means))
    
    mn40 = ShapeNet10(data_dir="dataset/PointDA_data/shapenet", split='test', num_points=1024)
    sonn = ScanObjectNNHardest(data_dir="dataset/scanobjectnn/h5_files/main_split", split='test', num_points=1024)
    i = 0
    item40 = mn40.__getitem__(i)
    while item40[2] != 3: # 35 is the index of 'toilet' in ModelNet40Ply2048.classes
        i += 1
        item40 = mn40.__getitem__(i)
        
    i = 0
    item10 = sonn.__getitem__(i)
    while item10[2] != 6: # 4 is the index of 'display' in sonn.classes
        i += 1
        item10 = sonn.__getitem__(i)
    item10 = sonn.__getitem__(i+1)
    print(item40[2], item10[2])
        
    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_subplot(projection='3d')
        
    ax.scatter(
        item10[0][:, 0], item10[0][:, 1], item10[0][:, 2],
        s=8.0, alpha=0.5, c='blue', cmap='Paired')
    
    ax.scatter(
        item40[0][:, 0], item40[0][:, 1], item40[0][:, 2],
        s=8.0, alpha=0.5, c='red', cmap='Paired')
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    # Show axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    plt.savefig("plot/bathtub_mn_sonn.png")

