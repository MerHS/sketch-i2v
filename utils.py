from pathlib import Path
import shutil
import pickle

import torch, torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np 
from random import randint

from tqdm import tqdm

class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    
    def __init__(self, args, model, optimizer, class_part_list, test_imgs, save_dir=None, save_freq=5):
        self.args = args
        self.model = model
        self.threshold = args.eval_threshold
        self.class_part_list = class_part_list
        self.class_len = sum(map(lambda x: len(x[1]), self.class_part_list))
        
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir()
        self.log_path = Path(save_dir) / 'loss_log.txt'
        self.save_freq = save_freq

        self.test_imgs = test_imgs
        self.epoch = 0
        self.loss_f = nn.BCELoss()

        if self.cuda:
            self.model = model.cuda()
            self.test_imgs = self.test_imgs.cuda()
            self.loss_f = self.loss_f.cuda()

        # assert self.vis.check_connection(), 'No connection could be formed quickly'

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        f1 = []

        estim_all = torch.zeros(self.class_len).long()
        per_class_tag_count = torch.zeros(self.class_len).long()
        true_positive = torch.zeros(self.class_len).long()
        if self.cuda:
            estim_all, per_class_tag_count, true_positive = \
                estim_all.cuda(), per_class_tag_count.cuda(), true_positive.cuda()

        for img_tensor, data_class in tqdm(data_loader, ncols=80):
            if self.cuda:
                img_tensor, data_class = img_tensor.cuda(), data_class.cuda()

            if is_train:
                self.optimizer.zero_grad()

            output = self.model(img_tensor)
            if self.args.vgg or self.args.resnet:
                output = F.sigmoid(output)

            loss = self.loss_f(output, data_class)
            loop_loss.append(loss.data.item() / len(data_loader))

            if is_train:
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                top_1_index = output.data.max(1)[1].view(-1, 1)
                accuracy.append((data_class.data.gather(1, top_1_index)).sum().item())

                estim_class = (output.data.view(output.shape[0], -1) >= self.threshold).long()
                data_c = data_class.data.long()
                
                true_positive += (data_c * estim_class).long().sum(0)
                per_class_tag_count += data_c.sum(0)
                estim_all += estim_class.sum(0)

        mode = "train" if is_train else "test"
        
        loss_value = sum(loop_loss)
        accuracy_value = sum(accuracy) / len(data_loader.dataset)
        loss_txt = f">>>[{mode} {self.epoch}] loss: {loss_value:.10f} / top-1 accuracy: {accuracy_value:.2%}"
        print(loss_txt)
        
        precision_per_class = true_positive.float() / estim_all.float()
        recall_per_class = true_positive.float() / per_class_tag_count.float()
        precision_per_class[torch.isnan(precision_per_class)] = 0
        recall_per_class[torch.isnan(recall_per_class)] = 0
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        f1_per_class[torch.isnan(f1_per_class)] = 0

        mx = precision_per_class.max()
        avg = precision_per_class.mean()
        per10 = (precision_per_class >= 0.1).sum() / (len(precision_per_class)) 
        ppc_txt = f"  precis per class / avg: {avg*100:5.3f}%, f1 >= 10%: {per10*100:5.3f}, max: {mx*100:5.3f}%"
        print(ppc_txt)

        mx = recall_per_class.max()
        avg = recall_per_class.mean()
        per10 = (recall_per_class >= 0.1).sum() / (len(recall_per_class)) 
        rpc_txt = f"  recall per class / avg: {avg*100:5.3f}%, f1 >= 10%: {per10*100:5.3f}, max: {mx*100:5.3f}%"
        print(rpc_txt)

        mx = f1_per_class.max()
        avg = f1_per_class.mean()
        per10 = (f1_per_class >= 0.1).sum() / (len(f1_per_class)) 
        f1_txt =  f"  f1     per class / avg: {avg*100:5.3f}%, f1 >= 10%: {per10*100:5.3f}, max: {mx*100:5.3f}%"
        print(f1_txt)
        
        with self.log_path.open('a') as f:
            f.write(loss_txt + '\n')
            f.write(ppc_txt + '\n')
            f.write(rpc_txt + '\n')
            f.write(f1_txt + '\n\n')
        
    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            self._iteration(data_loader)

    def test(self, data_loader, get_mask=False):
        self.model.eval()
        with torch.no_grad():
            self._iteration(data_loader, is_train=False)
            if get_mask:
                test_tensor = self.test_imgs.clone()
                mask = self.model.module.get_mask(test_tensor)
                mask = nn.functional.interpolate(mask, scale_factor=8)
                mask_tensor = torch.cat([self.test_imgs, mask], 1)
                b, c, h, w = mask_tensor.size()
                mask_tensor = mask_tensor.view(b*c, 1, h, w)
            else:
                mask_tensor = None

        return mask_tensor

    def loop(self, epochs, train_data, test_data, raw_data, scheduler=None, do_save=True):
        with self.log_path.open('a') as f:
            f.write(str(self.args) + '\n')
        
        for ep in range(self.args.resume_epoch + 1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            self.epoch = ep

            self.train(train_data)
            mask_tensor = self.test(test_data, get_mask=(not self.args.old))
            if not self.args.color:
                self.test(raw_data, get_mask=False)

            if do_save:
                self.save(ep, mask_tensor)

    def save(self, epoch, mask_tensor=None, **kwargs):
        if self.save_dir is not None:
            model_out_path = self.save_dir
            state = {
                "epoch": epoch, 
                "weight": self.model.state_dict(), 
                "optimizer": self.optimizer.state_dict()
            }
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / f"model_epoch_{epoch}.pth")

            if mask_tensor is not None:
                torchvision.utils.save_image(mask_tensor, model_out_path / f"mask_epoch_{epoch}.png",
                    nrow=7, padding=0)


def get_classid_dict(tag_dump_path):
    cv_dict = dict()
    iv_dict = dict()

    try:
        f = open(tag_dump_path, 'rb')
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        cv_tag_list = pkl['cv_tag_list']
        iv_part_list = pkl['iv_part_list']
        cv_part_list = pkl['cv_part_list']
    except EnvironmentError:
        raise Exception(f'{tag_dump_path} does not exist. You should make tag dump file using taglist/tag_indexer.py')

    for i, tag_id in enumerate(iv_tag_list):
        iv_dict[tag_id] = i
    for i, tag_id in enumerate(cv_tag_list):
        cv_dict[tag_id] = i

    return (iv_dict, cv_dict, iv_part_list, cv_part_list)


def read_tagline_txt(tag_txt_path, img_dir_path, classid_dict, class_len, data_size=0, read_all=False):
    # tag one-hot encoding + 파일 있는지 확인
    if not tag_txt_path.exists():
        raise Exception(f'tag list text file "{tag_txt_path}" does not exist.')

    tag_set = set(classid_dict.keys())
    tag_class_list = []
    file_id_list = []

    data_limited = data_size != 0
    count = 0

    with tag_txt_path.open('r') as f:
        for line in f:
            tag_list = list(map(int, line.split()))
            file_id = tag_list[0]
            tag_list = tag_list[1:]
            
            if not (img_dir_path / f'{file_id}.png').exists():
                continue

            if not read_all and len(tag_list) < 8:
                continue
            
            tag_class = torch.zeros(class_len, dtype=torch.float)

            tag_exist = False
            for tag in tag_list:
                if tag in tag_set:
                    try:
                        tag_class[classid_dict[tag]] = 1
                        tag_exist = True
                    except IndexError as e:
                        print(len(classid_dict), class_len, tag, classid_dict[tag])
                        raise e

            if not tag_exist:
                continue
                
            file_id_list.append(file_id)
            tag_class_list.append(tag_class)

            count += 1
            if data_limited and count > data_size:
                break

    print(f'file len {len(file_id_list)} class list len {len(tag_class_list)} class_len {class_len}')
    return (file_id_list, tag_class_list)
