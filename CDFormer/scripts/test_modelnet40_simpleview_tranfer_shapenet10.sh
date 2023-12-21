#!/bin/bash

set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore main_cls_simpleview_mn40_transfer.py --config config/modelnet40/modelnet40_simpleview_transfer_shapenet10.yaml \
                                 model_path /home/disi/CDFormer/output/modelnet40/cdformer/eval/2023-11-28/model/cdformer_last_mn40_mn40_simpleview.pth
