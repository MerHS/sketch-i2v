from pathlib import Path
import shutil
import pickle

import torch, torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np 
from random import randint
from itertools import accumulate

from tqdm import tqdm

class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    
    def __init__(self, args, model, optimizer, class_part_list, save_dir=None):
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
        
        self.epoch = 0
        self.loss_f = nn.BCELoss()

        if self.cuda:
            if model:
                self.model = model.cuda()
            self.loss_f = self.loss_f.cuda()

        # assert self.vis.check_connection(), 'No connection could be formed quickly'

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        data_len = len(data_loader.dataset)

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
        self.print_evaluation(mode, data_len, loop_loss, accuracy, estim_all, per_class_tag_count, true_positive)

    def print_evaluation(self, mode, data_len, loop_loss, accuracy, estim_all, per_class_tag_count, true_positive):
        loss_value = sum(loop_loss)
        accuracy_value = sum(accuracy) / data_len
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
        ppc_txt = f"  precis per class / avg: {avg*100:5.3f}%, >= 10%: {per10*100:5.3f}, max: {mx*100:5.3f}%"
        print(ppc_txt)

        mx = recall_per_class.max()
        avg = recall_per_class.mean()
        per10 = (recall_per_class >= 0.1).sum() / (len(recall_per_class)) 
        rpc_txt = f"  recall per class / avg: {avg*100:5.3f}%, >= 10%: {per10*100:5.3f}, max: {mx*100:5.3f}%"
        print(rpc_txt)

        mx = f1_per_class.max()
        avg = f1_per_class.mean()
        per10 = (f1_per_class >= 0.1).sum() / (len(f1_per_class)) 
        f1_txt =  f"  f1     per class / avg: {avg*100:5.3f}%, >= 10%: {per10*100:5.3f}, max: {mx*100:5.3f}%"
        print(f1_txt + '\n')
        
        with self.log_path.open('a') as f:
            f.write(loss_txt + '\n')
            f.write(ppc_txt + '\n')
            f.write(rpc_txt + '\n')
            f.write(f1_txt + '\n\n')
        
    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            self._iteration(data_loader)

    def test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            self._iteration(data_loader, is_train=False)

    def loop(self, epochs, train_data, test_data, raw_data, scheduler=None, do_save=True):
        with self.log_path.open('a') as f:
            f.write(str(self.args) + '\n')
        
        for ep in range(self.args.resume_epoch + 1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            self.epoch = ep

            self.train(train_data)
            self.test(test_data)
            if not self.args.color:
                self.test(raw_data)

            if do_save:
                self.save(ep)

    def save(self, epoch, **kwargs):
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


class GanTrainer(Trainer):
    def __init__(self, args, models, opts, class_part_list, test_imgs, save_dir=None):
        super(GanTrainer, self).__init__(args, None, None, class_part_list, save_dir)
        self.g_model, self.d_model = models
        self.g_opt, self.d_opt = opts
        self.test_imgs = test_imgs
        self.part_len = len(class_part_list)
        self.clen_list = list(map(lambda x: len(x[1]), class_part_list))
        self.clen_list.insert(0, 0)
        self.clen_list = np.array(list(accumulate(self.clen_list)))

        if self.cuda:
            self.g_model = self.g_model.cuda()
            self.d_model = self.d_model.cuda()
            self.test_imgs = test_imgs

    def _masking(self, mask, img_tensor, data_class, rand_idx=-1):
        img_len = img_tensor.shape[0]

        b, c, h, w = mask.shape
        b_ind = np.arange(img_len)
        if rand_idx == -1:
            mask_ind = np.random.randint(self.part_len, size=img_len)
        else:
            mask_ind = np.repeat(rand_idx, img_len)
            
        mask = mask[b_ind, mask_ind, :, :].unsqueeze(1)
        masked_img = 1 - (1 - img_tensor) * (1 - mask)

        clone_class = data_class.clone()
        if self.cuda:
            clone_class = clone_class.cuda()

        cmask_from, cmask_to = self.clen_list[mask_ind], self.clen_list[mask_ind+1]
        for i in range(data_class.shape[0]):
            clone_class[i, cmask_from[i]:cmask_to[i]] = 0

        return masked_img, clone_class

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        data_len = len(data_loader.dataset)

        estim_all = torch.zeros(self.class_len).long()
        per_class_tag_count = torch.zeros(self.class_len).long()
        true_positive = torch.zeros(self.class_len).long()
        if self.cuda:
            estim_all, per_class_tag_count, true_positive = \
                estim_all.cuda(), per_class_tag_count.cuda(), true_positive.cuda()

        # train
        for img_tensor, data_class in tqdm(data_loader, ncols=80):
            img_len = img_tensor.shape[0]
            y_real_, y_fake_ = torch.ones(img_len, 1), torch.zeros(img_len, 1)
            if self.cuda:
                img_tensor, data_class = img_tensor.cuda(), data_class.cuda()
                y_real_, y_fake_ = y_real_.cuda(), y_fake_.cuda()

            if is_train:
                # D
                self.d_opt.zero_grad()

                d_real_c, d_real_adv = self.d_model(img_tensor)
                d_real_loss = self.loss_f(d_real_c, data_class) + self.loss_f(d_real_adv, y_real_)

                d_mask = self.g_model(img_tensor)
                d_masked_img, d_masked_class = self._masking(d_mask, img_tensor, data_class)

                d_fake_c, d_fake_adv = self.d_model(d_masked_img)
                d_fake_loss = self.loss_f(d_fake_c, d_masked_class) + self.loss_f(d_fake_adv, y_fake_)

                d_loss = d_real_loss + d_fake_loss

                d_loss.backward()
                self.d_opt.step()
                d_loss_n = d_loss.data.item() / len(data_loader)

                # G
                self.g_opt.zero_grad()

                g_mask = self.g_model(img_tensor)

                g_masked_img, g_masked_class = self._masking(g_mask, img_tensor, data_class)
                g_class, g_adv = self.d_model(g_masked_img)

                g_loss = self.loss_f(g_class, g_masked_class) + self.loss_f(g_adv, y_real_)

                g_loss.backward()
                self.g_opt.step()
                g_loss_n = g_loss.data.item() / len(data_loader)

                loop_loss.append((d_loss_n, g_loss_n))
            else:
                d_real_c, d_real_adv = self.d_model(img_tensor)
                adv_loss = self.loss_f(d_real_adv, y_real_)
                class_loss = self.loss_f(d_real_c, data_class)
                loop_loss.append((adv_loss, class_loss))

            # evaluation
            with torch.no_grad():
                top_1_index = d_real_c.data.max(1)[1].view(-1, 1)
                accuracy.append((data_class.data.gather(1, top_1_index)).sum().item())

                estim_class = (d_real_c.data.view(d_real_c.shape[0], -1) >= self.threshold).long()
                data_c = data_class.data.long()
                
                true_positive += (data_c * estim_class).long().sum(0)
                per_class_tag_count += data_c.sum(0)
                estim_all += estim_class.sum(0)

        mode = "train" if is_train else "test"
        self.print_evaluation(mode, data_len, loop_loss, accuracy, estim_all, per_class_tag_count, true_positive)

    def print_evaluation(self, mode, data_len, loop_loss, accuracy, estim_all, per_class_tag_count, true_positive):
        d_loss, g_loss = [list(t) for t in zip(*loop_loss)]

        d_loss_value = sum(d_loss)
        if mode == "train":
            d_loss_txt = f">>>d_loss: {d_loss_value:.10f}"
        else:
            d_loss_txt = f">>>d_loss: {d_loss_value + sum(g_loss):.10f} / adv_loss: {d_loss_value:.10f} / below: class_loss"
        print(d_loss_txt)

        with self.log_path.open('a') as f:
            f.write(d_loss_txt + '\n')

        super(GanTrainer, self).print_evaluation(mode, data_len, g_loss, accuracy, estim_all, per_class_tag_count, true_positive)

    def test(self, data_loader):
        self.g_model.eval()
        self.d_model.eval()
        with torch.no_grad():
            self._iteration(data_loader, is_train=False)
            
    def train(self, data_loader):
        self.g_model.train()
        self.d_model.train()
        with torch.enable_grad():
            self._iteration(data_loader)

    def get_mask_img_set(self, img_tensor, data_class, mask_idx):
        self.g_model.eval()
        with torch.no_grad():
            g_mask = self.g_model(img_tensor)
            g_masked_img, _ = self._masking(g_mask, img_tensor, data_class, mask_idx)
        
        return g_mask, g_masked_img

    def loop(self, epochs, train_data, test_data, raw_data, scheduler=None, do_save=True):
        test_img, _ = test_data.__iter__().__next__()
        self.save_img("keras_sketch.png", test_img)
        if self.cuda:
            test_img = test_img.cuda()

        if not self.args.color:
            raw_img, _ = raw_data.__iter__().__next__()
            self.save_img("raw_sketch.png", raw_img)
            if self.cuda:
                raw_img = raw_img.cuda()


        with self.log_path.open('a') as f:
            f.write(str(self.args) + '\n')
        
        for ep in range(self.args.resume_epoch + 1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            self.epoch = ep

            self.train(train_data)
            self.test(test_data)
            if not self.args.color:
                self.test(raw_data)

            if do_save:
                self.save(ep)
                
                for part_i in range(self.part_len):
                    mask, mx_img = self.get_mask_img_set(test_img)
                    self.save_img(f'keras_mask_pt{part_i}_{ep}.png', mask)
                    self.save_img(f'keras_del_pt{part_i}_{ep}.png', mx_img)
                    if not self.args.color:
                        mask, mx_img = self.get_mask_img_set(raw_img)
                        self.save_img(f'raw_mask_pt{part_i}_{ep}.png', mask)
                        self.save_img(f'raw_del_pt{part_i}_{ep}.png', mx_img)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = self.save_dir
            state = {
                "epoch": epoch, 
                "g_model": self.g_model.state_dict(), 
                "d_model": self.d_model.state_dict(),
                "g_opt": self.g_opt.state_dict(),
                "d_opt": self.d_opt.state_dict()
            }
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / f"model_epoch_{epoch}.pth")

    def save_img(self, file_name, save_img):
        if self.save_dir is not None:
            model_out_path = self.save_dir
            torchvision.utils.save_image(save_img, model_out_path / file_name, nrow=7, padding=0)       


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
