from pathlib import Path

import pickle
import os.path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.se_resnet import se_resnext50
from model.vgg import vgg11_bn
from model.datasets import SketchDataset, ColorDataset
from utils import *
from tqdm import tqdm

from train import get_dataloader

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'taglist', 'tag_dump.pkl')

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--thread", default=8, type=int)
    p.add_argument("--vgg", action="store_true")
    p.add_argument("--gpu", default=1, type=int)
    p.add_argument("--train_file", default="model.pth")
    p.add_argument("--data_dir", default=DATA_DIRECTORY)
    p.add_argument("--tag_dump", default=TAG_FILE_PATH)
    p.add_argument("--data_size", default=200000, type=int)
    p.add_argument("--color", action="store_true")
    p.add_argument("--load_path", default="result.pth")

    # must set it true
    p.add_argument("--valid", default=True)

    args = p.parse_args()

    if not Path(args.file_name).exists():
        raise Exception(f"{args.file_name} does not exists.")

    with open(args.tag_dump, 'rb') as f:
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        cv_tag_list = pkl['cv_tag_list']
        tag_dict = pkl['tag_dict']
        tag_dict = {v: k for k, v in tag_dict.items()}

    with open(args.train_file, 'rb') as f:
        network_weight = torch.load(f)['weight']

    in_channels = 3 if args.color else 1
    tag_list = cv_tag_list if args.color else iv_tag_list

    class_len, train_loader, test_loader = get_dataloader(args)

    if args.vgg:
        network = vgg11_bn(num_classes=class_len, in_channels=1)
    else:
        network = se_resnext50(num_classes=class_len, input_channels=1)

    network.eval()
    with torch.no_grad():
        class_count = torch.zeros(class_len).float()
        score = torch.zeros(class_len).int()
        for img_tensor, data_class in tqdm(data_loader, ncols=80):
            if self.cuda:
                img_tensor, data_class = img_tensor.cuda(), data_class.cuda()

            output = network(img_tensor)
            
            estim_class = output >= 0.2
            score_tensor = estim_class[data_class.byte()].int().sum(0)
            score += score_tensor

            total_len += class_count

        accuracy = score.float() / class_count * 100
        
        with open('evaluate_result.txt', 'w') as f:
            for i in range(class_len):
                f.write(f'{tag_dict[tag_list[i]]:30s} {score[i]:6d} {total_len[i]:6d} {accuracy[i]:6.4f}\n')
