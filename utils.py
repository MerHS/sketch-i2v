from pathlib import Path
import torch
import numpy as np 

from tqdm import tqdm


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=5):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        for data, target in tqdm(data_loader, ncols=80):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data.item() / len(data_loader))
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        mode = "train" if is_train else "test"
        print(f">>>[{mode}] loss: {sum(loop_loss):.2f} / accuracy: {sum(accuracy) / len(data_loader.dataset):.2%}")
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

def get_tag_to_idx_dict(file_path):
    """
    file_path: pathlib.Path

    file format:
    342 # tag class length
    0 302 # dict_key tag_index
    1 24
    2 4254
    ...
    """
    if not file_path.exists():
        raise Exception(f'tag to index txt "{str(file_path)} does not exist."')
    
    tag_dict = dict()
    with file_path.open('r') as f:
        cls_len = int(f.readline())
        for line in f:


def read_tagline_txt(tag_txt_path, img_dir_path, tag_to_idx_dict, class_len):
    # tag one-hot encoding + 파일 있는지 확인
    if not tag_txt_path.exists():
        raise Exception(f'tag list text file "{tag_txt_path}" does not exist.')

    tag_set = set(tag_to_idx_dict.keys())
    tag_class_list = []
    file_id_list = []
    with tag_txt_path.open('r') as f:
        for line in f:
            tag_list = list(map(int, line.split(' ')))
            file_id = tag_list[0]
            tag_list = tag_list[1:]
            
            if not (img_dir_path / f'{file_id}.png').exists():
                continue
            
            tag_class = torch.zeros(class_len, dtype=torch.float)

            tag_exist = False
            for tag in tag_list:
                if tag in tag_set:
                    tag_class[tag_to_idx_dict[tag]] = 1
                    tag_exist = True

            if not tag_exist:
                continue
                
            file_id_list.append(file_id)
            tag_class_list.append(tag_class)

    return (file_id_list, tag_class_list)