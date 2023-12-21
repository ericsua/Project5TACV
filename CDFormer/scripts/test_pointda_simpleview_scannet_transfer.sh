#!/bin/bash

set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore main_cls_simpleview_pointda.py --config config/pointda/pointda_simpleview_scannet10.yaml debug 0 \
                                                            model_path /home/disi/CDFormer/output/pointda/scannet/2023-12-07-16-39-23/model/model_best.pth

#weight /home/disi/CDFormer/output/modelnet40/cdformer/2023-11-23-17-44-05/model/model_last.pth
