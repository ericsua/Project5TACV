"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import os
import sys
sys.path.append(os.getcwd())
import glob
import h5py
import numpy as np
import pickle
import logging
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from util.data_util import data_prepare_modelnet40 as data_prepare
import pdb
from util.modelnet40 import ModelNet40Ply2048
import matplotlib.pyplot as plt

def normalize_point_cloud(point_cloud):
    # Compute the mean for each dimension (X, Y, Z)
    mean = np.mean(point_cloud, axis=0)

    # Subtract the mean from each point
    normalized_point_cloud = point_cloud - mean

    # Compute the maximum distance from the origin
    #max_distance = np.max(np.linalg.norm(normalized_point_cloud, axis=1))
    max_distance = np.max(np.sqrt(np.sum(normalized_point_cloud[:, :3] ** 2, axis=1)))
    
    # Divide each point by the maximum distance
    #normalized_point_cloud /= max_distance
    normalized_point_cloud[:, :3] /= (max_distance + np.finfo(float).eps)

    return normalized_point_cloud

def load_data_graspnet(data_dir, partition, url, mode, sensor):
    all_data = []
    all_label = []
    for id, class_id in enumerate(sorted(glob.glob(os.path.join(data_dir, partition, mode, sensor, '*')))):
        print("file class", class_id, partition, id)
        for pointer in sorted(glob.glob(os.path.join(f'{class_id}', '*.xyz'))):
            #print("pointer", pointer, "id", id)
            # with h5py.File(h5_name, 'r') as f:
                # data = f['data'][:].astype('float32')
                # label = f['label'][:].astype('int64')
            # points = np.load(pointer, allow_pickle=True)
            # print("keys", points.tolist().keys())
            points = np.loadtxt(pointer, delimiter=' ')
            #print("points", points)
            points = points.reshape(-1, 3)
            #print("points", points.shape)
            #points = points[:, [0, 2, 1]]
            #points = normalize_point_cloud(points)
            #np.random.shuffle(points)
            all_data.append(points)
            #print("file npy", pointer, np.load(pointer).shape, "id", id)
            #all_data.append(pointer)
            all_label.append(id)
    all_data = np.array(all_data)
    print("all data", all_data.shape)
    all_label = np.array(all_label)
    print("all label", all_label.shape)
    return all_data, all_label

def load_data_modelnet10(data_dir, partition, url):
    all_data = []
    all_label = []
    for id, class_name in enumerate(sorted(glob.glob(os.path.join(data_dir, '*')))):
        #print("file class", class_name, partition, id)
        for pointer in sorted(glob.glob(os.path.join(f'{class_name}', f'{partition}', '*.npy'))):
            #print("pointer", pointer, "id", id)
            # with h5py.File(h5_name, 'r') as f:
                # data = f['data'][:].astype('float32')
                # label = f['label'][:].astype('int64')
            points = np.load(pointer)
            points = points[:, [0, 2, 1]]
            points = normalize_point_cloud(points)
            #np.random.shuffle(points)
            all_data.append(points)
            #print("file npy", pointer, np.load(pointer).shape, "id", id)
            #all_data.append(pointer)
            all_label.append(id)
    all_data = np.array(all_data)
    #print("all data", all_data.shape)
    all_label = np.array(all_label)
    #print("all label", all_label.shape)
    return all_data, all_label

def load_data_scannet10(data_dir, partition, url):
    all_data = []
    all_label = []
    #pointer = glob.glob('dataset/PointDA_data/scannet/test_0.h5')
    with h5py.File('dataset/PointDA_data/scannet/test_0.h5', 'r') as f:
        data = f['data'][:,:,:3].astype('float32')
        label = f['label'][:].astype('int64')
    
    print("data", data.shape)
    print("data 0", data[50].shape, data[50])
    print("label 0", label[50])
    print("label", label.shape)
    data = [normalize_point_cloud(d) for d in data]
    data = np.array(data)
    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        data[50][:, 0], data[50][:, 1], data[50][:, 2],
        s=8.0, alpha=0.5, c='red', cmap='Paired')
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    # Show axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    plt.savefig("plot/scannet.png")
    #data = data[:, :, [0, 2, 1]]
    all_data = np.array(all_data)
    #print("all data", all_data.shape)
    all_label = np.array(all_label)
    #print("all label", all_label.shape)
    return data, label
    # for id, class_name in enumerate(sorted(glob.glob(os.path.join(data_dir, '*')))):
    #     #print("file class", class_name, partition, id)
    #     for pointer in sorted(glob.glob(os.path.join(f'{class_name}', f'{partition}', '*.npy'))):
    #         #print("pointer", pointer, "id", id)
    #         # with h5py.File(h5_name, 'r') as f:
    #             # data = f['data'][:].astype('float32')
    #             # label = f['label'][:].astype('int64')
    #         points = np.load(pointer)
    #         points = points[:, [0, 2, 1]]
    #         points = normalize_point_cloud(points)
    #         #np.random.shuffle(points)
    #         all_data.append(points)
    #         #print("file npy", pointer, np.load(pointer).shape, "id", id)
    #         #all_data.append(pointer)
    #         all_label.append(id)
    all_data = np.array(all_data)
    #print("all data", all_data.shape)
    all_label = np.array(all_label)
    #print("all label", all_label.shape)
    return all_data, all_label
        

def load_data_single_file(data_dir, partition, url):    
    #pointer = glob.glob(os.path.join(f'dataset/PointDA_data/modelnet/bathtub/test/bathtub_0107.npy'))[0]
    pointer = glob.glob(os.path.join(f'dataset/PointDA_data/modelnet/plant/test/plant_0284.npy'))[0]
    print("pointer", pointer)
    points = np.load(pointer)
    points = normalize_point_cloud(points)
    #np.random.shuffle(points)
    all_data.append(points)
    #print("file npy", pointer, np.load(pointer).shape, "id", id)
    #all_data.append(pointer)
    all_label.append(0)
    #all_data = np.concatenate(all_data, axis=0)
    all_data = np.array(all_data)
    #print("all data", all_data.shape)
    all_label = np.array(all_label)
    #print("all label", all_label.shape)
    
    return all_data, all_label



class GraspNetPC10(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.
    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    dir_name = 'modelnet40_ply_hdf5_2048'
    md5 = 'c9ab8e6dfb16f67afdab25e155c79e59'
    url = f'https://shapenet.cs.stanford.edu/media/{dir_name}.zip'
    
    classes = [
        'box', 'can', 'banana', 'powerdrill', 'scissors', 
        'pear', 'dish', 'camel', 'mouse', 'shampoo'
        ]

    classes_common_sonn = [
        'box'
    ]

    def __init__(self,
                 data_dir="dataset/GraspNetPointClouds",
                 split='train',
                 num_points=1024,
                 transform=None,
                 mode="Synthetic",
                 sensor="kinect"
                 ):
        
        data_dir = os.path.join(
            os.getcwd(), data_dir) if data_dir.startswith('.') else data_dir
        self.partition = 'train' if split.lower() == 'train' or mode == "Synthetic" else 'test'  # val = test
        self.data, self.label = load_data_graspnet(data_dir, self.partition, self.url, mode=mode, sensor=sensor)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform

    def __getitem__(self, idx):
        coord = self.data[idx][:self.num_points]
        #xyz = np.load(self.data[idx])
        # coord = xyz[:self.num_points]
        label = self.label[idx]

        if self.partition == 'train':
            np.random.shuffle(coord)

        if self.transform is not None:
            coord, _ = self.transform(coord, coord.copy())  # (coord, feat)

        feat = torch.tensor(coord, dtype=torch.float)
        coord = torch.tensor(coord, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return coord, feat, label

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return np.max(self.label) + 1
    
def index_nested(nested_list, target):
    for i, element in enumerate(nested_list):
        if isinstance(element, list):
            if target in element:
                return i
        else:
            if element == target:
                return i
    return -1
    
    
if __name__ == '__main__':
    gp = GraspNetPC10(data_dir="dataset/GraspNetPointClouds", split='test', num_points=1024, transform=None, mode="Real", sensor="kinect")
    i = gp.__getitem__(0)
    print("i", i[0].shape, i[1].shape, i[2])
    print("data", i[0])
    
    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_subplot(projection='3d')
    
    item40 = i[0]
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
    
    plt.savefig("plot/bathtub_graspnet.png")
    
    # mn10 = ModelNet10(data_dir="dataset/PointDA_data/modelnet", split='test', num_points=1024, transform=None)
    # mn40Test = ModelNet40Ply2048(data_dir="dataset/modelnet40ply2048", split='test', num_points=1024, transform=None)
    # mn40Train = ModelNet40Ply2048(data_dir="dataset/modelnet40ply2048", split='train', num_points=1024, transform=None)
    
    # item10 = mn10.__getitem__(0)
    
    # for i in range(len(mn40Test.data)):
    #     item40 = mn40Test.__getitem__(i)
    #     mean = torch.mean(item40[0], dim=0)
    #     max_distance = np.max(np.linalg.norm(item40[0], axis=1))
    #     min_distance = np.min(np.linalg.norm(item40[0], axis=1))
    #     print("item40 mean", i, ":", mean)
    #     print("item40 max_distance", i, ":", max_distance)
    #     print("item40 min_distance", i, ":", min_distance)
    #     #print("item10", item10[0].shape, "item40", item40[0].shape, item40[1].shape, item40[2])
    #     if torch.equal(item40[0],item10[0]):
    #         print("found test", i)
    #         print("item10", item10[0], "item40", item40[0])
    #         break
        
    # for i in range(len(mn10.data)):
    #     item10 = mn10.__getitem__(i)
    #     mean = torch.mean(item10[0], dim=0)
    #     max_distance = np.max(np.linalg.norm(item10[0], axis=1))
    #     min_distance = np.min(np.linalg.norm(item10[0], axis=1))
    #     print("item10 mean", i, ":", mean)
    #     print("item10 max_distance", i, ":", max_distance)
    #     print("item10 min_distance", i, ":", min_distance)
        
    # # for i in range(len(mn40Train.data)):
    # #     item40 = mn40Train.__getitem__(i)
    # #     if torch.equal(item40[0], item10[0]):
    # #         print("found train", i)
    # #         print("item10", item10[0], "item40", item40[0])
    # #         break
    
    # print("not found")
    # print("item10", item10[0].shape, item10[1].shape, item10[2])
    # item10 = mn10.__getitem__(0) # bathtub_0107
    # #item40 = mn40Test.__getitem__(1083) # bathtub_0107
    # item40 = mn40Test.__getitem__(26) # "plant/plant_0284.ply"
    # #T = np.linalg.inv(item10[0]) @ item40[0]
    # print("item10", item10[0], item10[2])
    
    
    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_subplot(projection='3d')
    
    # Define the rotation angle in radians (90 degrees)
    theta = np.radians(-90)

    # Define the rotation matrix around the y-axis
    rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                [0, 1, 0],
                                [-np.sin(theta), 0, np.cos(theta)]])
    
    rotation_matrix_x = np.array([[ 1,      0,          0,      ],
                                    [ 0,  np.cos(theta), -np.sin(theta)],
                                    [ 0,  np.sin(theta),  np.cos(theta)]])
    
    rotation_matrix_z = np.array([[ np.cos(theta), -np.sin(theta), 0],
                                    [ np.sin(theta),  np.cos(theta), 0],
                                    [ 0,          0,      1]])
    # Apply the rotation to the point cloud
    #rotated_point_cloud = np.dot(item10[0], rotation_matrix.T)
    #item40 = (rotated_point_cloud, item40[1], item40[2])
    
    #item40 = (item40[0][:, [0, 2, 1]], item40[1], item40[2])
    item10 = (item10[0][:, [0, 2, 1]], item10[1], item10[2])
    #rotated_point_cloud = np.dot(item10[0], rotation_matrix_x.T)
    #theta = np.radians(0)
    #rotated_point_cloud = np.dot(rotated_point_cloud, rotation_matrix.T)
    #item10 = (rotated_point_cloud, item10[1], item10[2])
    
    # rotated_point_cloud = np.dot(item40[0], rotation_matrix.T)
    # item40 = (rotated_point_cloud, item40[1], item40[2])
    
    
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
    
    plt.savefig("plot/bathtub_0107.png")

    
    
    # print("item40", item40[0], item40[2])
    # #print("inverse transform", T)
        

