from pathlib import Path
import shutil
import pickle

import torch, torchvision
from torch import nn
import numpy as np 
from random import randint

from tqdm import tqdm

class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    
    def __init__(self, model, optimizer, test_imgs, save_dir=None, save_freq=5):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        self.test_imgs = test_imgs
        if not self.save_dir.exists():
            self.save_dir.mkdir()
        self.log_path = Path(save_dir) / 'loss_log.txt'
        self.save_freq = save_freq
        self.loss_f = nn.BCELoss().cuda()

        # assert self.vis.check_connection(), 'No connection could be formed quickly'

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []

        for img_tensor, data_class in tqdm(data_loader, ncols=80):
            if self.cuda:
                img_tensor, data_class = img_tensor.cuda(), data_class.cuda()

            if is_train:
                self.optimizer.zero_grad()

            output = self.model(img_tensor)
            
            loss = self.loss_f(output, data_class)

            loop_loss.append(loss.data.item() / len(data_loader))
            top_1_index = output.data.max(1)[1].view(-1, 1)

            accuracy.append((data_class.data.gather(1, top_1_index)).sum().item())

            if is_train:
                loss.backward()
                self.optimizer.step()

        mode = "train" if is_train else "test"
        loss_value = sum(loop_loss)
        accuracy_value = sum(accuracy) / len(data_loader.dataset)
        loss_txt = f">>>[{mode}] loss: {loss_value:.10f} / top-1 accuracy: {accuracy_value:.2%}"
        print(loss_txt)
        
        with self.log_path.open('a') as f:
            f.write(loss_txt + '\n')
        
        return loss_value, accuracy_value

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            loss, correct = self._iteration(data_loader)

        return loss, (correct * 100)

    def test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss, correct = self._iteration(data_loader, is_train=False)
            
            test_tensor = self.test_imgs.clone()
            mask = self.model.get_mask(test_tensor)
            mask_tensor = torch.cat([self.test_imgs, mask], 1)
            b, c, _, _ = mask_tensor.size()
            mask_tensor = mask_tensor.view(b*c, 1, -1, -1)

        return loss, (correct * 100), mask_tensor

    def loop(self, args, epochs, train_data, test_data, scheduler=None, do_save=True):
        arg_text = str(args)
        # self.vis.text(arg_text)
        for ep in range(args.resume_epoch + 1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            train_loss, train_correct = self.train(train_data)
            test_loss, test_correct, mask_tensor = self.test(test_data)
            

            if do_save:
                self.save(ep, mask_tensor)

    def save(self, epoch, mask_tensor, **kwargs):
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
            torchvision.utils.save_image(mask_tensor, model_out_path / f"mask_epoch_{epoch}.png",
                nrow=8,padding=0)


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
