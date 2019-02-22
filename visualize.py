# CAM(Class Activation Mapping) Visualization / modified https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py

from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2, torch
import os, pickle

from model.se_resnet import se_resnext50
from model.multi_se_resnext import multi_serx50
from utils import get_classid_dict

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
OUT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'result')
TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'taglist', 'tag_dump.pkl')

def load_weight(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(pretrained_dict)

def return_cam(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    print(h,w)
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = 1 - (cam / np.max(cam))
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return np.uint8(np.average(output_cam, 0))

def class_info(tag_dump_path):
    try:
        f = open(tag_dump_path, 'rb')
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        iv_part_list = pkl['iv_part_list']
        tag_dict = pkl['tag_dict']
    except EnvironmentError:
        raise Exception(f'{tag_dump_path} does not exist. You should make tag dump file using taglist/tag_indexer.py')

    tag_rev_dict = dict()
    idx_dict = dict()
    for k, v in tag_dict.items():
        tag_rev_dict[v] = k
    for i, tag_id in enumerate(iv_tag_list):
        idx_dict[tag_id] = i

    return iv_tag_list, idx_dict, iv_part_list, tag_rev_dict

def get_cam(args, model, tag_dump, features_blobs, weight_softmax):
    img_path = args.img_path
    img_pil = Image.open(img_path)

    preprocess = transforms.Compose([transforms.Resize((256, 256), interpolation=Image.LANCZOS),
           transforms.ToTensor()])

    img_tensor = preprocess(img_pil).unsqueeze(0)

    (tag_list, idx_dict, tag_part_list, tag_rev_dict) = tag_dump

    result = model(img_tensor)
    probs, idx = result.squeeze().sort(0, True)
    rs = result.squeeze()
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], tag_rev_dict[tag_list[idx[i]]]))

    part_no = args.part
    parts = tag_part_list[part_no][1]

    high_parts = [high_i for high_i in idx[:20] if tag_list[high_i] in parts]
    for hi in high_parts:
        print(f"higher: {rs[hi]:4f} / {tag_rev_dict[tag_list[hi]]}")

    # generate class activation mapping for the top1 prediction
    print(f'get avg of {tag_part_list[part_no][0]}')
    # CAMs = return_cam(features_blobs[0], weight_softmax, high_parts)
    CAMs = return_cam(features_blobs[0], weight_softmax, [tag_part_list[part_no[0]]])

    # render the CAM and output
    print(f'output CAM.jpg for the top1 prediction: {tag_rev_dict[tag_list[idx[0]]]}')
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs, (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5

    return result

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--load_path", default="result.pth")
    p.add_argument("--tag_dump", default=TAG_FILE_PATH)
    p.add_argument("--old", action="store_true")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("img_path")
    args = p.parse_args()

    tag_dump = class_info(args.tag_dump)
    tag_list, idx_dict, tag_part_list, tag_rev_dict = tag_dump
    class_len = len(tag_list)
    in_channels = 1

    if args.old:
        model = se_resnext50(num_classes=class_len, input_channels=in_channels)
    else:
        model = multi_serx50(class_list=tag_part_list, input_channels=in_channels)

    with open(args.load_path, 'rb') as f:
        loaded = torch.load(f)
        network_weight = loaded['weight']
        if 'optimizer' in loaded:
            opt_weight = loaded['optimizer']
        else:
            opt_weight = None
        load_weight(model, network_weight)

    model.eval()
    finalconv_name = 'layer4'
    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    with torch.no_grad():
        result = get_cam(args, model, tag_dump, features_blobs, weight_softmax)
    cv2.imwrite("test.png", result)

#python visualize.py 1077425.png --load_path=./result_dir2/model_epoch_92.pth --old