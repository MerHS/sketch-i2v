import argparse
import pickle
import torch
import cv2

from torch import nn
from PIL import Image
from torchvision import transforms
from sketchify.crop import make_256px_square
from sketchify.sketchify import get_sketch
from model.se_resnet import se_resnext50

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
    p.add_argument("--out", default="out.png")
    p.add_argument("--sketch", action='store_true')
    args = p.parse_args()

    with open('taglist/tag_dump.pkl', 'rb') as f:
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        tag_dict = pkl['tag_dict']
        tag_dict = {v: k for k, v in tag_dict.items()}

    with open(args.train_file, 'rb') as f:
        network_weight = torch.load(f)['weight']

    network = se_resnext50(num_classes=len(iv_tag_list), input_channels=1)
    load_weight(network, network_weight)

    network.eval()

    img = cv2.imread(args.file_name)
    img, _, _ = make_256px_square(img, extend=(True, True))
    cv2.imshow("main", img)
    if args.sketch:
        sketch_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        sketch_img = get_sketch(img)
        cv2.imshow("sketch", sketch_img)

    pil_img = Image.fromarray(sketch_img)
    norm_img = to_normalized_tensor(pil_img)
    norm_img = norm_img.view(-1, 1, 256, 256)

    classifier = nn.Sigmoid()

    with torch.no_grad():
        class_vec = network(norm_img)
        class_vec = classifier(class_vec).view(-1)

        props = []
        for i, prop in enumerate(class_vec):
            if prop > 0.20:
                props.append((iv_tag_list[i], prop.item()))

        props.sort(key=lambda x:x[1], reverse=True)
        for tag_key, prop in props:
            print(tag_dict[tag_key], prop)

        cv2.waitKey(0)