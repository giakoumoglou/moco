#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=64gb:ngpus=4
#PBS -lwalltime=72:00:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate torch

python main_moco.py -a resnet50 --lr 0.03 --epochs 200 --batch-size 256 --dist-url 'tcp://localhost:10001' --save-dir ./output/moco_r50_200ep/ --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/
	
python main_lincls.py -a resnet50 --lr 30.0 --batch-size 256 --pretrained ./output/moco_r50_200ep/checkpoint_0199.pth.tar --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/