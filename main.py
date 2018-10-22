import json 
import pickle
import os
from pathlib import Path

import cv2
import numpy as np
import sketchify
import crop

# directory path of 512px or original folder of danbooru2017 dataset
IMAGE_DIRECTORY = 'F:\\IMG\\danbooru\\danbooru2017\\512px' 

TAG_MONOCHROME = 1681
TAG_GREYSCALE = 513837
TAG_COMIC = 63
AVAILABLE_EXT = ['jpeg', 'jpg', 'bmp', 'png']

image_dir_path = Path(IMAGE_DIRECTORY)
if not image_dir_path.exists():
    raise Exception('Directory of image "'+ IMAGE_DIRECTORY + '" does not exists.')


def get_image_path(id):
    image_path = image_dir_path / f'{(id%1000):04}' / f'{id}.jpg'
    return image_path

def get_image(id, aspect):
    image_path = get_image_path(id)
    img = crop.get_cropped_image(image_path, aspect)
    return img

def metadata_to_tagline(metadata):
    id_list = map(lambda tag: tag['id'], metadata['tags'])
    return metadata['id'] + ' ' + ' '.join(id_list)

def move_lineart(id, aspect, metadata):
    img = get_image(id, aspect)
    if img is not None:
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def move_color_image(id, aspect, metadata):
    img = get_image(id, aspect)
    if img is not None:
        return

    sketch = sketchify.get_sketch(img)


def show_image(id, aspect):
    img = get_image(id, aspect)
    if img is not None:
        cv2.imshow('original', img)
        cv2.imshow('keras-high-intensity', sketchify.get_keras_high_intensity(img))
        skt = sketchify.get_sketch(img)
        cv2.imshow('sketch',skt)
        cv2.waitKey(0)
    else:
        print(f'cannot find {id}.jpg')


if __name__ == '__main__':
    count = 0
    for json_file_name in os.listdir('./metadata'):
        break
        if count > 0: 
            break
        count += 1

        if not json_file_name.endswith('.json'):
            continue

        with open('metadata/' + json_file_name, 'r', encoding='utf8') as f:
            for line in f:
                try: 
                    metadata = json.loads(line)
                    tags = metadata['tags']

                    # use image only
                    if metadata['file_ext'] not in AVAILABLE_EXT:
                        continue

                    # use safe only
                    if metadata['rating'] is not 's':
                        continue

                    tag_id_list = list(map(lambda t: int(t['id']), tags))
                    height, width = int(metadata['image_height']), int(metadata['image_width'])
                    aspect = width / height

                    # drop too long or small image 
                    if height < 512 and width < 512:
                        continue
                    if not ((2/3) <= aspect <= (3/2)):
                        continue

                    # drop comic
                    if TAG_COMIC in tag_id_list:
                        continue

                    # lineart는 따로 처리 (monochrome or greyscale)
                    if TAG_MONOCHROME in tag_id_list or TAG_GREYSCALE in tag_id_list:
                        move_lineart(id, aspect, metadata)
                    else:
                        move_color_image(id, aspect, metadata)

                except KeyError as e:
                    print(e)

# img = get_image(263167, 551/778)
# if img is not None:
#     cv2.imshow('original', img)
#     cv2.imshow('keras', sketchify.get_keras_high_intensity(img))
#     skt = sketchify.get_sketch(img)
#     cv2.imshow('name',skt)
#     cv2.waitKey(0)
