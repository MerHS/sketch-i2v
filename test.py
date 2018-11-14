import argparse
import pickle
import torch
import cv2

from pathlib import Path
from torch import nn
from PIL import Image
from torchvision import transforms
from sketchify.crop import make_square
from model.se_resnet import se_resnext50
from model.vgg import vgg11_bn

to_normalized_tensor = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.9184], std=[0.1477])])

def load_weight(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(pretrained_dict)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("file_name")
    p.add_argument("--train_file", default="model.pth")
    p.add_argument("--sketch", action='store_true')
    p.add_argument("--color", action='store_true')
    p.add_argument("--blend", action='store_true')
    p.add_argument("--show", action='store_true')
    p.add_argument("--vgg", action='store_true')
    args = p.parse_args()

    if not Path(args.file_name).exists():
        raise Exception(f"{args.file_name} does not exists.")

    with open('taglist/tag_dump.pkl', 'rb') as f:
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        cv_tag_list = pkl['cv_tag_list']
        tag_dict = pkl['tag_dict']
        tag_dict = {v: k for k, v in tag_dict.items()}

    with open(args.train_file, 'rb') as f:
        network_weight = torch.load(f)['weight']

    in_channels = 3 if args.color else 1
    tag_list = cv_tag_list if args.color else iv_tag_list
    if args.vgg:
        network = vgg11_bn(num_classes=len(tag_list), in_channels=in_channels)
    else:
        network = se_resnext50(num_classes=len(tag_list), input_channels=in_channels)
    load_weight(network, network_weight)

    network.eval()

    img = cv2.imread(args.file_name)
    img, _, _ = make_square(img, size=512, extend=(True, True))
    if args.show:
        cv2.imshow("main", img)
    
    if args.color:
        sketch_img = img
    elif args.sketch:
        sketch_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif args.blend:
        from sketchify.sketchify import get_sketch
        sketch_img = get_sketch(img, blend=0.15)
        if args.show:
            cv2.imshow("sketch", sketch_img)
    else:
        from sketchify.sketchify import get_keras_high_intensity
        sketch_img = get_keras_high_intensity(img, 1.4)
        if args.show:
            cv2.imshow("sketch", sketch_img)
    
    sketch_img = cv2.resize(sketch_img, (256, 256), interpolation=cv2.INTER_AREA)

    pil_img = Image.fromarray(sketch_img)
    norm_img = to_normalized_tensor(pil_img)
    norm_img = norm_img.view(-1, in_channels, 256, 256)

    with torch.no_grad():
        class_vec = network(norm_img).view(-1)

        props = []
        for i, prop in enumerate(class_vec):
            if prop >= 0.2:
                props.append((tag_list[i], prop.item()))

        props.sort(key=lambda x:x[1], reverse=True)
        for tag_key, prop in props:
            print(tag_dict[tag_key], prop)
        if args.show:
            cv2.waitKey(0)
