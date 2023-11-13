#!/bin/bash

# training models with our datr
config='datr_vitb_256_mae_ce_32x4_ep300'
python tracking/train.py --script ostrack --config ${config} --save_dir /path/of/model --mode multiple --nproc_per_node 4 --datr 1
python tracking/test.py ostrack ${config} --dataset lasot_extension_subset --threads 4 --num_gpus 4


