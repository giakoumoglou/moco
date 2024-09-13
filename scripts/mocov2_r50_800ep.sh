#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=64gb:ngpus=4
#PBS -lwalltime=72:00:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate torch

python main_moco.py -a resnet50 --lr 0.03 --epochs 800 --batch-size 256 --mlp --moco-t 0.2 --aug-plus --cos --dist-url 'tcp://localhost:10001' --save-dir ./output/mocov2_r50_800ep/ --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/
	
python main_lincls.py -a resnet50 --lr 30.0 --batch-size 256 --pretrained ./output/mocov2_r50_800ep/checkpoint_0799.pth.tar --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/