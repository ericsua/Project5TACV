#!/bin/bash

set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore main_cls_simpleview_sonn_transfer.py --config config/scanobjectnn/scanobjectnn_cdformer_simpleview_transfer_mn40.yaml \
                                 model_path /home/disi/CDFormer/output/scanobjectnn/cdformer/2023-11-24-18-42-46/model/model_last.pth
