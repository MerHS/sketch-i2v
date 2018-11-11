from pathlib import Path

import pickle
import os.path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.se_resnet import se_resnext50
from model.datasets import SketchDataset
from utils import *

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
OUT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'result')
TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'taglist', 'tag_dump.pkl')

def get_dataloader(args):
    batch_size = args.batch_size
    data_dir = args.data_dir

    to_normalized_tensor = [transforms.ToTensor(), transforms.Normalize(mean=[0.9184], std=[0.1477])]
    data_augmentation = [transforms.RandomHorizontalFlip(), ]

    data_dir_path = Path(data_dir)
    
    train_dir = data_dir_path / ("train" if not args.valid else "validation")
    test_dir = data_dir_path / "test"
    
    iv_dict, cv_dict = get_classid_dict(args.tag_dump)
    class_len = len(iv_dict.keys())

    test_size = args.data_size // 10 if not args.valid else args.data_size

    print('reading train set tagline')
    (train_id_list, train_iv_class_list) = read_tagline_txt(
        train_dir / "tags.txt", train_dir, iv_dict, class_len, args.data_size)
    print('reading test set tagline')
    (test_id_list, test_iv_class_list) = read_tagline_txt(
        test_dir / "tags.txt", test_dir, iv_dict, class_len, test_size)

    print('making train dataset...')
    
    train = SketchDataset(train_dir, train_id_list, train_iv_class_list, override_len=args.data_size,
        transform = transforms.Compose(data_augmentation + to_normalized_tensor))
    test = SketchDataset(test_dir, test_id_list, test_iv_class_list, override_len=test_size,
        transform = transforms.Compose(to_normalized_tensor), is_train=False)
    
    print('making dataloader...')
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=args.thread)
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=args.thread)
    
    return class_len, train_loader, test_loader


def main(args):
    print('making dataloader...')
    class_len, train_loader, test_loader = get_dataloader(args)
    gpu_count = args.gpu if args.gpu > 0 else 1
    gpus = list(range(torch.cuda.device_count()))
    gpus = gpus[:gpu_count]

    model = se_resnext50(num_classes=class_len, input_channels=1)

    if args.resume_epoch != 0:
        with open(args.load_path, 'rb') as f:
            network_weight = torch.load(f)['weight']
        model_dict = model.state_dict()
        pretrained_dict = {k[7:]: v for k, v in network_weight.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(pretrained_dict)

    se_resnet = nn.DataParallel(model, device_ids=gpus)
    optimizer = optim.SGD(params=se_resnet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=args.lr_gamma)

    print(f'training params: {args}')
    print('setting trainer...')
    trainer = Trainer(se_resnet, optimizer, save_dir=args.out_dir)

    print(f'start loop')
    trainer.loop(args, args.epoch, train_loader, test_loader, scheduler, do_save=(not args.valid))


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epoch", default=100, type=int)
    p.add_argument("--thread", default=8, type=int)
    p.add_argument("--gpu", default=1, type=int)
    p.add_argument("--lr", default=0.2, type=float)
    p.add_argument("--lr_gamma", default=0.15, type=float)
    p.add_argument("--lr_step", default=30, type=int)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--decay", default=0.0001, type=float)
    p.add_argument("--data_dir", default=DATA_DIRECTORY)
    p.add_argument("--out_dir", default=OUT_DIRECTORY)
    p.add_argument("--tag_dump", default=TAG_FILE_PATH)
    p.add_argument("--data_size", default=200000, type=int)
    p.add_argument("--valid", action="store_true")
    p.add_argument("--resume_epoch", default=0, type=int)
    p.add_argument("--load_path", default="result.pth")
    
    args = p.parse_args()

    main(args)
    # calculate(args)
