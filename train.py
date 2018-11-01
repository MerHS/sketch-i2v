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
from utils import Trainer, read_tagline_txt

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')

def get_classid_dict():
    tagid_to_classid_dict = dict()
    iv_tag_list = None
    with open('taglist/tag_dump.pkl', 'rb') as f:
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
    
    for i, tag_id in enumerate(iv_tag_list):
        tagid_to_classid_dict[tag_id] = i

    return tagid_to_classid_dict

# TODO: calculate mean & std
def get_dataloader(batch_size, image_dir):
    to_normalized_tensor = [transforms.ToTensor(), transforms.Normalize(mean=[0.9184], std=[0.1477])]
    data_augmentation = [transforms.RandomHorizontalFlip(), ]

    image_dir_path = Path(image_dir)
    
    train_dir = image_dir_path / "train"
    test_dir = image_dir_path / "test"
    
    classid_dict = get_classid_dict()
    class_len = len(classid_dict.keys())

    (train_id_list, train_class_list) = read_tagline_txt(train_dir / "tags.txt", train_dir, classid_dict, class_len)
    (test_id_list, test_class_list) = read_tagline_txt(test_dir / "tags.txt", test_dir, classid_dict, class_len)

    train = SketchDataset(train_dir, train_id_list, train_class_list, 
        transform = transforms.Compose(data_augmentation + to_normalized_tensor))
    test = SketchDataset(test_dir, test_id_list, test_class_list, 
        transform = transforms.Compose(to_normalized_tensor))

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=6)
    
    return class_len, train_loader, test_loader

def main(args):
    class_len, train_loader, test_loader = get_dataloader(args.batch_size, args.image_dir)
    gpus = list(range(torch.cuda.device_count()))
    se_resnet = nn.DataParallel(se_resnext50(num_classes=class_len, input_channels=1),
                                device_ids=gpus)

    optimizer = optim.SGD(params=se_resnet.parameters(), lr=0.6 / 1024 * args.batch_size, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)

    trainer = Trainer(se_resnet, optimizer, save_dir=args.out_dir)
    trainer.loop(args.epoch, train_loader, test_loader, scheduler)

def calculate(args):
    _, train_loader, _ = get_dataloader(args.batch_size, args.image_dir)
    count = 0
    means = []
    stds = []
    iter = 0
    print('start')
    for t, c in train_loader:
        t = t.cuda()
        batch_list = t.reshape(args.batch_size, -1)
        means.append(batch_list.mean())
        stds.append(batch_list.std(1).mean())
        count += 1
        if count % 100 == 0:
            iter += 1
            print(iter, torch.Tensor(means).mean(), torch.Tensor(stds).mean())
            if iter == 100:
                break

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epoch", default=100, type=int)
    p.add_argument("--data_dir", default=DATA_DIRECTORY)
    p.add_argument("--out_dir", default="./result")
    args = p.parse_args()

    main(args)
    # calculate(args)
