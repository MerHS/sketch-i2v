import argparse
import pickle
import torch
import cv2
import os.path

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

DATA_DIRECTORY = "/SSD/hyunsu/dataset"

data_dir = Path(DATA_DIRECTORY)

IMAGE_DIRECTORY = data_dir / "rgb_test"
TAG_FILE_PATH = data_dir / "tags.txt"

print("TAG_FILE_PATH ", TAG_FILE_PATH)
print("IMAGE_DIRECTORY ", IMAGE_DIRECTORY)

def load_weight(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(pretrained_dict)

def print_image_tags(tag_txt_path, img_dir_path, img_path, tag_dict_reverse, cvt_tag_list):

    img_id = os.path.basename(img_path)
    img_id = int(img_id[:-4])

    with tag_txt_path.open('r') as f:
        for line in f:
            tag_list = list(map(int, line.split()))
            file_id = tag_list[0]

            if img_id == file_id:
                tag_list = tag_list[1:]

               

                cvt_tags = []
                ivt_tags = []
                for tag_key in tag_list:
                    try:
                        if tag_key in cvt_tag_list:
                            cvt_tags.append(tag_dict_reverse[tag_key])
                        else:
                            ivt_tags.append(tag_dict_reverse[tag_key])
                    except:
                        print("there is wrong tag ", tag_key)
                
                print("cvt original tags ", ' '.join(cvt_tags))
                print("ivt original tags ", ' '.join(ivt_tags))


                print("================")
                break
            else:
                continue



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("file_name")
    p.add_argument("--train_file", default="model.pth")
    p.add_argument("--sketch", action='store_true')
    p.add_argument("--color", action='store_true')
    p.add_argument("--blend", action='store_true')
    p.add_argument("--show", action='store_true')
    p.add_argument("--vgg", action='store_true')
    p.add_argument("--metric", type=float, default=0.2)
    args = p.parse_args()

    if not Path(args.file_name).exists():
        raise Exception(f"{args.file_name} does not exists.")

    with open('taglist/tag_dump.pkl', 'rb') as f:
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        cv_tag_list = pkl['cv_tag_list']
        tag_dict = pkl['tag_dict']
        tag_dict_reverse = {v: k for k, v in tag_dict.items()}

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

    print(in_channels, args.color, len(tag_list))
    print_image_tags(TAG_FILE_PATH, IMAGE_DIRECTORY, args.file_name, tag_dict_reverse, tag_list)

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
            if prop >= args.metric:
                props.append((tag_list[i], prop.item()))

        props.sort(key=lambda x:x[1], reverse=True)
        for tag_key, prop in props:
            print(tag_dict_reverse[tag_key], prop)
        if args.show:
            cv2.waitKey(0)

