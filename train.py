from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.se_resnet import se_resnet50
from model.datasets import SketchDataset
from utils import Trainer

# TODO: calculate mean & std
def get_dataloader(batch_size, root):
    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    data_augmentation = [transforms.RandomHorizontalFlip()]

    root_path = Path(root)
    train_dir = root_path / "train"
    test_dir = root_path / "test"

    train_tag_file = train_dir / "tags.txt"
    test_tag_file = test_dir / "tags.txt"
    tag_index_file = root_path / "tag_index.txt"


    train = SketchDataset(train_dir, file_id_list, tag_class_list, 
        transform = transforms.Compose(data_augmentation + to_normalized_tensor))
    test = SketchDataset(test_dir, file_id_list, tag_class_list, 
        transform = transforms.Compose(to_normalized_tensor))

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_loader, test_loader

def main(args):
    train_loader, test_loader = get_dataloader(args.batch_size, args.root)
    gpus = list(range(torch.cuda.device_count()))
    se_resnet = nn.DataParallel(se_resnet50(num_classes=args.num_classes),
                                device_ids=gpus)
    optimizer = optim.SGD(params=se_resnet.parameters(), lr=0.6 / 1024 * args.batch_size, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    trainer = Trainer(se_resnet, optimizer, F.cross_entropy, save_dir=".")
    trainer.loop(args.epoch, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", help="image data root")
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--epoch", default=150, type=int)
    p.add_argument("--num_classes", default=1000, type=int)
    args = p.parse_args()
    main(args)