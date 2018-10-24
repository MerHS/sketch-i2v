from pathlib import Path

import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.se_resnet import se_resnext50
from model.datasets import SketchDataset
from utils import Trainer, read_tagline_txt

IMAGE_DIRECTORY = 'F:\\IMG\\danbooru\\danbooru2017\\dataset'

def get_classid_dict():
    tagid_to_classid_dict = dict()
    iv_tag_list = None
    with open('taglist/tag_dump.pkl', 'rb') as f:
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
    
    class_len = len(iv_tag_list)
    for i, tag_id in enumerate(iv_tag_list):
        tagid_to_classid_dict[tag_id] = i

    return tagid_to_classid_dict

# TODO: calculate mean & std
def get_dataloader(batch_size, image_dir):
    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    data_augmentation = [transforms.RandomHorizontalFlip()]

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
        train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_loader, test_loader

def main(args):
    train_loader, test_loader = get_dataloader(args.batch_size, args.root)
    gpus = list(range(torch.cuda.device_count()))
    se_resnet = nn.DataParallel(se_resnext50(num_classes=args.num_classes),
                                device_ids=gpus)

    optimizer = optim.SGD(params=se_resnet.parameters(), lr=0.6 / 1024 * args.batch_size, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)

    trainer = Trainer(se_resnet, optimizer, F.cross_entropy, save_dir="./")
    trainer.loop(args.epoch, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--epoch", default=150, type=int)
    p.add_argument("--num_classes", default=1000, type=int)
    p.add_argument("--image_dir", default=IMAGE_DIRECTORY)
    p.add_argument("--out_dir", default="./result")
    args = p.parse_args()

    main(args)