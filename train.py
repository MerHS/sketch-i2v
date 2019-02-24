from pathlib import Path

import pickle, math, random
import os.path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as tvF

from model.se_resnet import se_resnext50
from model.multi_se_resnext import multi_serx50
from torchvision.models.vgg import vgg16_bn
from torchvision.models.resnet import resnet50
from model.datasets import MultiImageDataset, RawSketchDataset
from utils import *
from test import load_weight
from tqdm import tqdm
from PIL import Image

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
OUT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'result')
TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'taglist', 'tag_dump.pkl')

def rot_crop(x):
    """return maximum width ratio of rotated image without letterbox"""
    x = abs(x)
    deg45 = math.pi * 0.25
    deg135 = math.pi * 0.75
    x = x * math.pi / 180
    a = (math.sin(deg135 - x) - math.sin(deg45 - x))/(math.cos(deg135-x)-math.cos(deg45-x))
    return math.sqrt(2) * (math.sin(deg45-x) - a*math.cos(deg45-x)) / (1-a)

class RandomFRC(transforms.RandomResizedCrop):
    """RandomHorizontalFlip + RandomRotation + RandomResizedCrop 2 images"""
    def __call__(self, img):
        if random.random() < 0.5:
            img = tvF.hflip(img)
        if random.random() < 0.5:
            rot = random.uniform(-10, 10)
            crop_ratio = rot_crop(rot)
            img = tvF.rotate(img, rot, resample=Image.BILINEAR)
            img = tvF.center_crop(img, int(img.size[0] * crop_ratio))
        
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        # return the image with the same transformation
        return (tvF.resized_crop(img, i, j, h, w, self.size, self.interpolation))

def get_dataloader(args):
    batch_size = args.batch_size
    data_dir = Path(args.data_dir)
    
    iv_dict, cv_dict, iv_part_list, cv_part_list = get_classid_dict(args.tag_dump)
    
    data_augmentation = [RandomFRC(512, scale=(0.9, 1.0), ratio=(0.95, 1.05))]
    to_tensor = [transforms.Resize((256, 256), interpolation=Image.LANCZOS),
                 transforms.ToTensor()]
    
    if args.color:
        train_dir_list = [data_dir / 'rgb_train']
        test_dir = data_dir / "rgb_test"        
        class_dict = cv_dict
        class_part_list = cv_part_list
    else:
        train_dir_list = [data_dir / 'keras_train', data_dir / 'simpl_train', data_dir / 'xdog_train']
        test_dir = data_dir / "keras_test"
        class_dict = iv_dict
        class_part_list = iv_part_list
        data_augmentation.append(transforms.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.2))

    test_raw_dir = data_dir / "liner_test"
    class_len = len(class_dict)

    print('reading train set tagline')
    (train_id_list, train_class_list) = read_tagline_txt(
        data_dir / "tags.txt", train_dir_list[0], class_dict, class_len, args.data_size, read_all=True)

    print('reading test set tagline')
    (test_id_list, test_class_list) = read_tagline_txt(
        data_dir / "tags.txt", test_dir, class_dict, class_len, read_all=True)

    print('making train dataset...')
    
    train = MultiImageDataset(train_dir_list, train_id_list, train_class_list, override_len=args.data_size,
        transform=transforms.Compose(data_augmentation + to_tensor), is_color=args.color)

    to_tensor = transforms.Compose(to_tensor)
    if args.color:
        test_img_dir = test_dir
        test_raw = None
    else:
        test_img_dir = test_raw_dir
        (test_raw_id_list, test_raw_class_list) = read_tagline_txt(
            data_dir / "tags.txt", test_raw_dir, class_dict, class_len, read_all=True)
        test_raw = RawSketchDataset(test_raw_dir, test_raw_id_list, test_raw_class_list, 
            transform=to_tensor)
    
    test = MultiImageDataset([test_dir], test_id_list, test_class_list, transform=to_tensor, is_color=args.color)

    conv_arg = 'RGB' if args.color else 'L'
    test_imgs = []
    count = 0
    for fn in test_img_dir.iterdir():
        test_img = Image.open(str(fn)).convert(conv_arg)
        test_imgs.append(to_tensor(test_img))

        count += 1
        if count > 16:
            break
    test_imgs = torch.stack(test_imgs)
            
    print(f'test_imgs size: {test_imgs.size()}')
    print('making dataloader...')
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=args.thread)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=args.thread)
    if args.color:
        raw_loader = None
    else:
        raw_loader = DataLoader(test_raw, batch_size=batch_size, shuffle=True, num_workers=args.thread)

    return class_len, train_loader, test_loader, raw_loader, class_part_list, test_imgs


def main(args):
    print('making dataloader...')
    class_len, train_loader, test_loader, raw_loader, iv_part_list, test_imgs = get_dataloader(args)
    gpu_count = args.gpu if args.gpu > 0 else 1
    gpus = list(range(torch.cuda.device_count()))
    gpus = gpus[:gpu_count]
    opt_weight = None

    in_channels = 3 if args.color else 1
    if args.vgg:
        model = vgg16_bn(num_classes=class_len)
    elif args.resnet:
        model = resnet50(num_classes=class_len)
    elif args.old:
        model = se_resnext50(num_classes=class_len, input_channels=in_channels)
    else:
        model = multi_serx50(class_list=iv_part_list, input_channels=in_channels)

    if args.resume_epoch != 0:
        with open(args.load_path, 'rb') as f:
            loaded = torch.load(f)
            network_weight = loaded['weight']
            if 'optimizer' in loaded:
                opt_weight = loaded['optimizer']
            else:
                opt_weight = None
        load_weight(model, network_weight)
        last_epoch = args.resume_epoch
    else:
        last_epoch = -1

    model_par = nn.DataParallel(model, device_ids=gpus)
    optimizer = optim.SGD(params=model_par.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    if opt_weight:
        load_weight(optimizer, opt_weight)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=args.lr_gamma, last_epoch=last_epoch)

    print(f'training params: {args}')
    print('setting trainer...')
    trainer = Trainer(model_par, optimizer, save_dir=args.out_dir, test_imgs=test_imgs)

    print(f'start loop')
    trainer.loop(args, args.epoch, train_loader, test_loader, raw_loader, scheduler, do_save=True)


def calc_meanstd(args):
    pass
    # class_len, train_loader, test_loader = get_dataloader(args)
    # mean = 0.
    # std = 0.
    # nb_samples = 0.
    # for data, _ in tqdm(train_loader, ncols=80):
    #     batch_samples = data.size(0)
    #     data = data.view(batch_samples, data.size(1), -1)
    #     mean += data.mean(2).sum(0)
    #     std += data.std(2).sum(0)
    #     nb_samples += batch_samples

    # mean /= nb_samples
    # std /= nb_samples
    # print(f"mean: {mean} / std: {std}")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epoch", default=100, type=int)
    p.add_argument("--thread", default=8, type=int)
    p.add_argument("--vgg", action="store_true")
    p.add_argument("--resnet", action="store_true")
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
    p.add_argument("--resume_epoch", default=0, type=int)
    p.add_argument("--load_path", default="result.pth")
    p.add_argument("--color", action="store_true")
    p.add_argument("--old", action="store_false")
    p.add_argument("--calc", action="store_true")

    args = p.parse_args()
    print(args)

    if args.calc:
        calc_meanstd(args)
    else:
        main(args)

#python train.py --data_dir=../dataset --out_dir ./result_dir2 --batch_size=48 --thread=4 --color
