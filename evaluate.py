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
from test import load_weight

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
    p.add_argument("--data_size", default=0, type=int)
    p.add_argument("--color", action="store_true")
    p.add_argument("--metric", default=0.2, type=float)

    # must set it True and False
    p.add_argument("--valid", type=bool, default=True)
    p.add_argument("--calc", type=bool, default=False)

    args = p.parse_args()
    print(args)

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

    class_len, valid_loader, _ = get_dataloader(args, read_all=True)

    if args.vgg:
        network = vgg11_bn(num_classes=class_len, in_channels=1)
    else:
        network = se_resnext50(num_classes=class_len, input_channels=1)
    
    load_weight(network, network_weight)

    if args.gpu > 0:
        network = network.cuda()

    network.eval()
    with torch.no_grad():
        class_count = torch.zeros(class_len).long()
        score = torch.zeros(class_len).long()
        total_count = 0
        for img_tensor, data_class in tqdm(valid_loader, ncols=80):
            if args.gpu > 0:
                img_tensor, data_class = img_tensor.cuda(), data_class.cuda()
            output = network(img_tensor)
            output = output.view(output.shape[0], -1)
            if args.gpu > 0:
                output = output.cpu()
                data_class = data_class.cpu()
            estim_class = output >= args.metric
            data_class = data_class.long()
            
            score_tensor = (data_class * estim_class.long()).long().sum(0)
            score += score_tensor
            class_count += data_class.sum(0)
            total_count += data_class.shape[0]
        
        accuracy = score.float() / class_count.float() * 100
        
        result = []
        for i in range(class_len):
            tag = tag_dict[tag_list[i]]
            scr = score[i]
            count = class_count[i]
            acc = accuracy[i]
            result.append((acc, count, scr, tag))
        result.sort(reverse=True)

        with open('evaluate_result.txt', 'w') as f:
            f.write(f'total {total_count} imgs / metric: {args.metric}\n')
            for acc, count, score, tag in result:
                f.write(f'{tag:25s} {score:6d} {count:6d} {acc:8.4f}\n')
        print(f'finished! total {total_count} imgs / check evalute_result.txt')

