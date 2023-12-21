#!/bin/bash

set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore main_cls_simpleview_graspnet.py --config config/graspnet/graspnet_simpleview_synth_kinect.yaml debug 0 \
                                                            model_path /home/disi/CDFormer/output/graspnet/synth/kinect/2023-12-08-16-00-58/model/model_best.pth

#weight /home/disi/CDFormer/output/modelnet40/cdformer/2023-11-23-17-44-05/model/model_last.pth
