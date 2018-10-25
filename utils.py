from pathlib import Path
import shutil

import torch
from torch import nn
import numpy as np 
from random import randint

from tqdm import tqdm


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, save_dir=None, save_freq=5):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []

        for img_tensor, data_class in tqdm(data_loader, ncols=80):
            if self.cuda:
                img_tensor, data_class = img_tensor.cuda(), data_class.cuda()

            if is_train:
                self.optimizer.zero_grad()

            output = self.model(img_tensor)
            loss_f = nn.BCEWithLogitsLoss().cuda()
            loss = loss_f(output, data_class)

            loop_loss.append(loss.data.item() / len(data_loader))
            top_1_index = output.data.max(1)[1].view(-1, 1)

            accuracy.append((data_class.data.gather(1, top_1_index)).sum().item())

            if is_train:
                loss.backward()
                self.optimizer.step()

        mode = "train" if is_train else "test"

        print(f">>>[{mode}] loss: {sum(loop_loss):.10f} / top-1 accuracy: {sum(accuracy) / len(data_loader.dataset):.2%}")
        
        return loop_loss, accuracy

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            loss, correct = self._iteration(data_loader)

    def test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss, correct = self._iteration(data_loader, is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            self.train(train_data)
            self.test(test_data)
            if ep % self.save_freq:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch, "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "model_epoch_{}.pth".format(epoch))


def read_tagline_txt(tag_txt_path, img_dir_path, classid_dict, class_len):
    # tag one-hot encoding + 파일 있는지 확인
    if not tag_txt_path.exists():
        raise Exception(f'tag list text file "{tag_txt_path}" does not exist.')

    tag_set = set(classid_dict.keys())
    tag_class_list = []
    file_id_list = []

    with tag_txt_path.open('r') as f:
        for line in f:
            tag_list = list(map(int, line.split(' ')))
            file_id = tag_list[0]
            tag_list = tag_list[1:]
            
            if not (img_dir_path / f'{file_id}.png').exists():
                continue

            if len(tag_list) < 8:
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

    return (file_id_list, tag_class_list)

def random_move(tag_txt_path, img_dir_path, dataset_dir_path):
    train_path = dataset_dir_path / 'train'
    test_path = dataset_dir_path / 'test'
    vaild_path = dataset_dir_path / 'validation'
    paths = [str(train_path), str(train_path), str(train_path), str(test_path), str(vaild_path)]

    for p in [test_path, train_path, vaild_path]:
        if not p.exists():
            p.mkdir()
    
    file_id_list = []
    with tag_txt_path.open('r') as f:
        for line in f:
            file_id = line.split()[0]
            if not ((img_dir_path / f'{file_id}.png').exists() and 
                (img_dir_path / f'{file_id}_sk.png').exists()):
                continue
            file_id_list.append(file_id)

    for file_id in file_id_list:
        file_path = img_dir_path / f'{file_id}.png'
        skt_path = img_dir_path / f'{file_id}_sk.png'
        n = randint(0, 4)

        shutil.move(str(file_path), paths[n])
        shutil.move(str(skt_path), paths[n])
        
