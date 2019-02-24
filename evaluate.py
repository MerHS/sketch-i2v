import pickle
import os.path
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.se_resnet import se_resnext50
from model.vgg import vgg11_bn
from model.datasets import MultiImageDataset
from utils import *
from train import get_dataloader
from test import load_weight

import imageio
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'taglist', 'tag_dump.pkl')

def get_dataset(args):
    batch_size = args.batch_size
    data_dir = Path(args.data_dir)
    
    iv_dict, cv_dict, iv_part_list, cv_part_list = get_classid_dict(args.tag_dump)
    to_tensor = [transforms.Resize((256, 256), interpolation=Image.LANCZOS),
                 transforms.ToTensor()]
    
    if args.color:
        test_dir = data_dir / "rgb_test"        
        class_dict = cv_dict
    else:
        test_dir = data_dir / "keras_test"
        class_dict = iv_dict

    class_len = len(class_dict)
    (test_id_list, test_class_list) = read_tagline_txt(
        data_dir / "tags.txt", test_dir, class_dict, class_len, read_all=True)

    to_tensor = transforms.Compose(to_tensor)

    test = MultiImageDataset([test_dir], test_id_list, test_class_list, transform=to_tensor, is_color=args.color)
    return test

def get_testloader(args, dataset):
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.thread)
    return test_loader

def get_network(args, model_path, class_len):
    with open(model_path, 'rb') as f:
        network_weight = torch.load(f)['weight']

    in_channels = 3 if args.color else 1

    if args.vgg:
        network = vgg11_bn(num_classes=class_len, in_channels=in_channels)
    else:
        network = se_resnext50(num_classes=class_len, input_channels=in_channels)
    
    load_weight(network, network_weight)

    if args.gpu > 0:
        network = network.cuda()

    network.eval()
    return network

def calculate(args, network, data_loader, tag_list, tag_dict):
    """return precision_per_class, recall_per_class, precision_all, recall_all, per_class_tag_count, img_count"""
    class_len = len(tag_list)

    with torch.no_grad():
        img_count = 0
        per_class_tag_count = torch.zeros(class_len).long()
        estim_all = torch.zeros(class_len).long()
        tp_all = torch.zeros(class_len).long()
        
        true_positive = torch.zeros(class_len).long()
        
        for img_tensor, data_class in tqdm(data_loader, ncols=80):
            if args.gpu > 0:
                img_tensor, data_class = img_tensor.cuda(), data_class.cuda()
            output = network(img_tensor)
            output = output.view(output.shape[0], -1)

            if args.gpu > 0:
                output = output.cpu()
                data_class = data_class.cpu()

            estim_class = (output >= args.threshold).long()
            data_class = data_class.long()

            true_positive += (data_class * estim_class).long().sum(0)

            per_class_tag_count += data_class.sum(0)
            estim_all += estim_class.sum(0)
            tp_all = true_positive.sum(0)

            img_count += data_class.shape[0]
        
    precision_per_class = true_positive.float() / estim_all.float()
    recall_per_class = true_positive.float() / per_class_tag_count.float()

    correct_count = float(true_positive.sum())
    precision_all = correct_count / estim_all.sum()
    recall_all = correct_count / per_class_tag_count.sum()

    result = {
        'precision_per_class' : precision_per_class,
        'recall_per_class': recall_per_class,
        'precision_all' : precision_all,
        'recall_all': recall_all,
        'per_class_tag_count': per_class_tag_count,
        'true_positive': true_positive,
        'img_count': img_count
    }

    return result

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--thread", default=8, type=int)
    p.add_argument("--vgg", action="store_true")
    p.add_argument("--gpu", default=1, type=int)
    p.add_argument("--save_path", default="evaluate")
    p.add_argument("--train_file", default="model.pth")
    p.add_argument("--train_dir", default="")
    p.add_argument("--data_dir", default=DATA_DIRECTORY)
    p.add_argument("--tag_dump", default=TAG_FILE_PATH)
    p.add_argument("--color", action="store_true")
    p.add_argument("--threshold", default=0.2, type=float)
    
    # do not change it 
    p.add_argument("--data_size", default=1, type=int)

    args = p.parse_args()
    print(args)

    with open(args.tag_dump, 'rb') as f:
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        cv_tag_list = pkl['cv_tag_list']
        tag_dict = pkl['tag_dict']
        tag_dict = {v: k for k, v in tag_dict.items()}

    tag_list = cv_tag_list if args.color else iv_tag_list
    class_len = len(tag_list)

    TEST_DIR_PATH = "F:\\새 폴더\\result_dir2"
    train_dir = TEST_DIR_PATH # args.train_dir
    
    dataset = get_dataset(args)

    precision_per_tag = list()
    recall_per_tag = list()
    precision_all = list()
    recall_all = list()

    if train_dir == "":
        model_path = args.train_file    
        network = get_network(args, model_path, class_len)
        data_loader = get_testloader(args, dataset)

        result = calculate(args, network, data_loader, tag_list, tag_dict)
        
        precision_per_tag.append(result['precision_per_class'])
        recall_per_tag.append(result['recall_per_tag'])
        precision_all.append(result['precision_all'])
        recall_all.append(result['recall_all'])
    else: 
        for epoch in range(1, 100):
            model_path = Path(train_dir) / f'model_epoch_{epoch}.pth'
            if not model_path.exists():
                break
            
            network = get_network(args, str(model_path), class_len)
            data_loader = get_testloader(args, dataset)

            result = calculate(args, network, data_loader, tag_list, tag_dict)

            precision_per_tag.append(result['precision_per_class'])
            recall_per_tag.append(result['recall_per_tag'])
            precision_all.append(result['precision_all'])
            recall_all.append(result['recall_all'])

    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    
    for ep, (pre_tag, rec_tag, pre_all, rec_all) in \
            enumerate(zip(precision_per_tag, recall_per_tag, precision_all, recall_all)):
        epoch = ep + 1
        save_name = Path(args.train_file).stem if train_dir == '' else f'{epoch:02d}_epoch'

        # precision per tag + precision > 10% count
        precision_list = []
        for i, prec in enumerate(pre_tag):
            tag = tag_dict[tag_list[i]]
            precision_list.append((prec, tag))
        precision_list.sort(reverse=True)

        file_path = str(save_path / (save_name + '-precision_tag.png'))    
        [pre_x, pre_y] = list(zip(*(precision_list[:100])))
        # TODO: make histogram
        
        # recall per tag + recall > 10% count
        recall_list = []
        for i, rec in enumerate(rec_tag):
            tag = tag_dict[tag_list[i]]
            recall_list.append((rec, tag))
        recall_list.sort(reverse=True)

        file_path = str(save_path / (save_name + '-recall_tag.png'))
        [pre_x, pre_y] = list(zip(*(recall_list[:100])))
        # TODO: make histogram

        # precision / recall all
        file_path = str(save_path / (save_name + '-pr_all.png'))

        fig, ax = plt.subplots()
        axes = plt.gca()
        axes.set_ylim([0, 1.])
        
        x_val = list(range(1, len(precision_all)))
        ax.plot(x_val, precision_all, label='precision')
        ax.plot(x_val, recall_all, label='recall')
        legend = ax.legend(loc='upper right', shadow=True)
        # legend.get_frame().set_facecolor('C0')
        ax.suptitle(f'Precision - Recall For All Classes ({save_name})' )
        ax.ylabel('Precision/Recall (All Classes)')
        
        plt.savefig(fig)

    # TODO: make mov
    # images = []
    # for e in range(num):
    #     img_name = path + '_epoch%03d_G_f_' % (start_num+e+1) + '.png'
    #     images.append(imageio.imread(img_name))
    # imageio.mimsave(path + '_G_f_generate_animation.gif', images, fps=5)