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

#from scanobjectnn import ScanObjectNNHardest
from matplotlib import pyplot as plt

def download_and_extract_archive(url, path, md5=None):
    # Works when the SSL certificate is expired for the link
    path = Path(path)
    extract_path = path
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / Path(url).name
        if not file_path.exists() or not check_integrity(file_path, md5):
            print(f'{file_path} not found or corrupted')
            print(f'downloading from {url}')
            context = ssl.SSLContext()
            with urllib.request.urlopen(url, context=context) as response:
                with tqdm(total=response.length) as pbar:
                    with open(file_path, 'wb') as file:
                        chunk_size = 1024
                        chunks = iter(lambda: response.read(chunk_size), '')
                        for chunk in chunks:
                            if not chunk:
                                break
                            pbar.update(chunk_size)
                            file.write(chunk)
            extract_archive(str(file_path), str(extract_path))
    return extract_path


def load_data(data_dir, partition, url):
    download_and_extract_archive(url, data_dir)
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition))):
        print(h5_name, partition)
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0).squeeze(-1)
    print(all_data.shape, all_label.shape)
    return all_data, all_label


class ModelNet40Ply2048(Dataset):
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
    classes = ['airplane',
               'bathtub',
               'bed',
               'bench',
               'bookshelf',
               'bottle',
               'bowl',
               'car',
               'chair',
               'cone',
               'cup',
               'curtain', 'desk', 'door',
               'dresser',
               'flower_pot',
               'glass_box',
               'guitar',
               'keyboard',
               'lamp',
               'laptop',
               'mantel',
               'monitor',
               'night_stand',
               'person',
               'piano',
               'plant',
               'radio',
               'range_hood',
               'sink',
               'sofa',
               'stairs',
               'stool',
               'table',
               'tent',
               'toilet',
               'tv_stand',
               'vase',
               'wardrobe',
               'xbox']
    
    classes_common_sonn = [
    ['dresser', 'wardrobe'], ['bench', 'chair', 'stool'], 'desk', 'monitor',
    'door', 'bookshelf', 'table', 'bed', 'sink', 'sofa', 'toilet']
    
    classes_common_pointda = [
        'bathtub', 'bed', 'bookshelf', ['dresser', 'wardrobe'], 'chair', 
        'lamp', 'monitor', 'plant', 'sofa', 'table'
    ]

    def __init__(self,
                 data_dir="dataset/modelnet40ply2048",
                 split='train',
                 num_points=1024,
                 transform=None
                 ):
        data_dir = os.path.join(
            os.getcwd(), data_dir) if data_dir.startswith('.') else data_dir
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.label = load_data(data_dir, self.partition, self.url)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform

    def __getitem__(self, idx):
        coord = self.data[idx][:self.num_points]
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
    
class ModelNet40Ply2048_sonn(ModelNet40Ply2048):
    def __init__(self, data_dir="dataset/modelnet40ply2048", split='train', num_points=1024, transform=None):
        super().__init__(data_dir, split, num_points, transform)
        
        filtered_points = list()
        filtered_labels = list()
        for pts, lbl in zip(self.data, self.label):
            index = index_nested(self.classes_common_sonn, self.classes[lbl])
            if index != -1: # if class is in common with mn40, no nested needed since we only have one level for sonn
            #if index_nested(self.classes_common_mn40, self.classes[lbl]) != -1 : # if class is in common with mn40
                filtered_points.append(pts)
                # get corresponding label index in sonn common classes
                #idx = index_nested(ModelNet40Ply2048.classes_common_sonn, self.classes[lbl])
                #idx = self.classes_common_mn40.index(self.classes[lbl])
                filtered_labels.append(index) # not the final label of mn40, but the index of common classes (could be a list), in test, we need to convert it to the final label of mn40
            #else:
            #    raise ValueError(f'1 Class {self.classes[lbl]} not found in ModelNet40Ply2048.classes_common_sonn')
                
        self.data = np.array(filtered_points)
        self.label = np.array(filtered_labels)
    
    def __getitem__(self, idx):
        coord = self.data[idx][:self.num_points]
        label = self.label[idx]

        if self.partition == 'train':
            np.random.shuffle(coord)

        if self.transform is not None:
            coord, _ = self.transform(coord, coord.copy())  # (coord, feat)


        height = coord[:, 1:2]
        height = height - height.min()

        feat = np.concatenate((coord, height), axis=1)
        feat = torch.tensor(feat, dtype=torch.float)
        #feat = torch.tensor(coord, dtype=torch.float)
        coord = torch.tensor(coord, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return coord, feat, label
        
        

if __name__ == "__main__":
    mn40 = ModelNet40Ply2048(data_dir="dataset/modelnet40ply2048", split='test', num_points=1024)
    sonn = ScanObjectNNHardest(data_dir="dataset/scanobjectnn/h5_files/main_split", split='test', num_points=1024)
    i = 0
    item40 = mn40.__getitem__(i)
    while item40[2] != 13: # 35 is the index of 'toilet' in ModelNet40Ply2048.classes
        i += 1
        item40 = mn40.__getitem__(i)
        
    i = 0
    item10 = sonn.__getitem__(i)
    while item10[2] != 3: # 4 is the index of 'display' in sonn.classes
        i += 1
        item10 = sonn.__getitem__(i)
        
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