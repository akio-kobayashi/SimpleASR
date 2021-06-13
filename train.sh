#!/bin/sh

python3 train.py --data data_40.h5 --train train.keys --valid valid.keys \
	--output model --out-stats stats.h5 --batch-size 5 --epochs 200 --learning-rate 1.0e-4
