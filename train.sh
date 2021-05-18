#!/bin/sh

python3 train.py --data data.h5 --train train.keys --valid valid.keys \
	--output model.h5 --out-stats stats.h5 --batch-size 25 --epochs 200 --learning-rate 1.0e-4

