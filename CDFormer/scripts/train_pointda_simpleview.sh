#!/bin/bash

set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore main_cls_simpleview_pointda.py --config config/pointda/pointda_simpleview.yaml debug 0 #weight /home/disi/CDFormer/output/modelnet40/cdformer/2023-11-23-17-44-05/model/model_last.pth
