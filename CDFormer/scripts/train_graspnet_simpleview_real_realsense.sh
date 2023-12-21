#!/bin/bash

set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore main_cls_simpleview_graspnet.py --config config/graspnet/graspnet_simpleview_real_realsense.yaml debug 0 weight /home/disi/CDFormer/output/graspnet/real/realsense/2023-12-08-00-21-03/model/model_last.pth
