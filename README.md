## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

This is a PyTorch implementation of the [MoCo paper](https://arxiv.org/abs/1911.05722):

```
@article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```

It also includes the implementation of the [MoCo v2 paper](https://arxiv.org/abs/2003.04297):

```
@article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```

### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on that code. Check the modifications by:
```
diff main_moco.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```

### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:

```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:

```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

### Models

Our pre-trained ResNet-50 models can be downloaded as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">mlp</th>
<th valign="bottom">aug+</th>
<th valign="bottom">cos</th>
<th valign="bottom">top-1 acc.</th>
<th valign="bottom">model</th>
<th valign="bottom">md5</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://arxiv.org/abs/1911.05722">MoCo v1</a></td>
<td align="center">200</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">60.6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>b251726a</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">67.7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>59fd9945</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">800</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">71.1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>a04e12f8</tt></td>
</tr>
</tbody></table>


### Transferring to Object Detection

See [./detection](detection).


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
