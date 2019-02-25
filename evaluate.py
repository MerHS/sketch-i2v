import pickle, math
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
        test_dir = data_dir / "liner_test"
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
        gpu_count = args.gpu if args.gpu > 0 else 1
        gpus = list(range(torch.cuda.device_count()))
        gpus = gpus[:gpu_count]
        network = nn.DataParallel(network, device_ids=gpus)
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

            img_count += data_class.shape[0]
        
    precision_per_class = true_positive.float() / estim_all.float()
    recall_per_class = true_positive.float() / per_class_tag_count.float()

    correct_count = float(true_positive.sum())
    precision_all = correct_count / float(estim_all.sum())
    recall_all = correct_count / float(per_class_tag_count.sum())

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

    # TEST_DIR_PATH = "F:\\새 폴더\\result_dir2"
    train_dir = args.train_dir
    
    dataset = get_dataset(args)

    precision_per_class = list()
    recall_per_class = list()
    precision_all = list()
    recall_all = list()

    if train_dir == "":
        model_path = args.train_file    
        network = get_network(args, model_path, class_len)
        data_loader = get_testloader(args, dataset)

        result = calculate(args, network, data_loader, tag_list, tag_dict)
        
        precision_per_class.append(result['recall_per_class'])
        recall_per_class.append(result['recall_per_class'])
        precision_all.append(result['precision_all'])
        recall_all.append(result['recall_all'])
    else: 
        for epoch in range(1, 101):
            model_path = Path(train_dir) / f'model_epoch_{epoch}.pth'
            if not model_path.exists():
                break
            
            network = get_network(args, str(model_path), class_len)
            data_loader = get_testloader(args, dataset)

            result = calculate(args, network, data_loader, tag_list, tag_dict)

            precision_per_class.append(result['precision_per_class'])
            recall_per_class.append(result['recall_per_class'])
            precision_all.append(result['precision_all'])
            recall_all.append(result['recall_all'])

    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    
    pre_all_list = list()
    rec_all_list = list()

    for ep, (pre_tag, rec_tag, pre_all, rec_all) in \
            enumerate(zip(precision_per_class, recall_per_class, precision_all, recall_all)):
        epoch = ep + 1
        save_name = Path(args.train_file).stem if train_dir == '' else f'{epoch:02d}_epoch'

        # precision per tag + precision >= 10% count
        precision_list = []
        precision_count = (pre_tag >= 0.1).sum()
        pre_percentage = precision_count.item() / len(pre_tag)
        for i, prec in enumerate(pre_tag):
            if math.isnan(prec):
                prec = 0
            tag = tag_dict[tag_list[i]]
            precision_list.append((prec, tag))
        precision_list.sort(reverse=True)

        file_path = str(save_path / (f'precision_class-{save_name}.png'))    
        [pre_y, pre_x] = list(zip(*(precision_list[:40])))
        # print(len(pre_x), pre_y)

        fig, ax = plt.subplots()
        plt.xticks(fontsize=7, rotation=90)
        ax.set_ylim([0, 1.])
        ax.bar(pre_x, pre_y)
        
        plt.xlabel(f'Precision >= 10% : {pre_percentage*100:5.3f}%')
        plt.ylabel(f'Precision (Per Classes) threshold {args.threshold}')
        plt.title(f'Precision Per Classes ({save_name})' )
        plt.subplots_adjust(bottom=0.30)

        fig.savefig(file_path)

        # recall per tag + recall >= 10% count
        recall_list = []
        recall_count = (rec_tag >= 0.1).sum()
        rec_percentage = recall_count.item() / len(rec_tag)
        for i, rec in enumerate(rec_tag):
            if math.isnan(rec):
                rec = 0
            tag = tag_dict[tag_list[i]]
            recall_list.append((rec, tag))
        recall_list.sort(reverse=True)

        file_path = str(save_path / (f'recall_class-{save_name}.png'))
        [rec_y, rec_x] = list(zip(*(recall_list[:40])))
        # print(len(rec_x), rec_y)

        fig, ax = plt.subplots()
        plt.xticks(fontsize=7, rotation=90)
        ax.set_ylim([0, 1.])
        ax.bar(rec_x, rec_y)
        
        plt.xlabel(f'Recall >= 10% : {rec_percentage*100:5.3f}%')
        plt.ylabel(f'Recall (Per Classes) threshold {args.threshold}')
        plt.title(f'Recall Per Classes ({save_name})' )
        plt.subplots_adjust(bottom=0.30)

        fig.savefig(file_path)

        pre_all_list.append(pre_all)
        rec_all_list.append(rec_all)

        plt.close('all')

    # precision / recall all
    file_path = str(save_path / 'precision_recall_all.png')

    fig, ax = plt.subplots()

    x_val = list(range(1, len(pre_all_list) + 1))
    ax.plot(x_val, pre_all_list, label='precision')
    ax.plot(x_val, rec_all_list, label='recall')
    print(pre_all_list, rec_all_list)

    legend = ax.legend(loc='upper right', shadow=True)
    # legend.get_frame().set_facecolor('C0')
    ax.set_ylim([0, 1.])
    plt.ylabel(f'Precision/Recall (All Classes) threshold {args.threshold}')

    plt.title(f'Precision - Recall For All Classes ({save_name})' )
    fig.savefig(file_path)

    # TODO: make mov
    # images = []
    # for e in range(num):
    #     img_name = path + '_epoch%03d_G_f_' % (start_num+e+1) + '.png'
    #     images.append(imageio.imread(img_name))
    # imageio.mimsave(path + '_G_f_generate_animation.gif', images, fps=5)
