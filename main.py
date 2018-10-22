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
OUTPUT_DIRECTORY = 'F:\\IMG\\danbooru\\danbooru2017\\256px'

TAG_MONOCHROME = 1681
TAG_GREYSCALE = 513837

# comic, photo, subtitled, english
TAG_BLACKLIST = [63, 4751, 12650, 172609]

AVAILABLE_EXT = ['jpeg', 'jpg', 'bmp', 'png']

image_dir_path = Path(IMAGE_DIRECTORY)
output_dir_path = Path(OUTPUT_DIRECTORY)

if not image_dir_path.exists():
    raise Exception('Directory of image "'+ IMAGE_DIRECTORY + '" does not exists.')

output_dir_path.mkdir(exist_ok=True)

def get_image_path(file_id):
    image_path = image_dir_path / f'{(int(file_id) % 1000):04}' / f'{file_id}.jpg'
    return image_path

def is_image_exists(file_id):
    image_path = get_image_path(file_id)
    return image_path.exists()

def get_cropped_image(file_id, aspect):
    image_path = get_image_path(file_id)
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    return crop.crop_square_image(img, aspect)

def metadata_to_tagline(metadata):
    id_list = map(lambda tag: tag['id'], metadata['tags'])
    return metadata['id'] + ' ' + ' '.join(id_list) + '\n'

def move_lineart(file_id, aspect, dest_path):
    img = get_cropped_image(file_id, aspect)
    if img is None:
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    square_img, _, _ = crop.make_256px_square(img)

    cv2.imwrite(str(dest_path / f'{file_id}.png'), square_img)


def move_color_image(file_id, aspect, dest_path):
    img = get_cropped_image(file_id, aspect)
    if img is None:
        return

    sketch = sketchify.get_sketch(img)
    square_img, cropped, extend = crop.make_256px_square(img)
    square_sketch, _, _ = crop.make_256px_square(sketch, cropped, extend)

    cv2.imwrite(str(dest_path / f'{file_id}.png'), square_img)
    cv2.imwrite(str(dest_path / f'{file_id}_sk.png'), square_sketch)


def show_image(file_id, aspect):
    img = get_cropped_image(file_id, aspect)
    if img is not None:
        cv2.imshow('original', img)
        cv2.imshow('keras-high-intensity', sketchify.get_keras_high_intensity(img))
        skt = sketchify.get_sketch(img)
        cv2.imshow('sketch',skt)
        cv2.waitKey(0)
    else:
        print(f'cannot find {file_id}.jpg')


if __name__ == '__main__':
    count = 0
    lineart_dir = output_dir_path / 'lineart'
    lineart_dir.mkdir(exist_ok=True)

    for json_file_name in os.listdir('./metadata'):

        if not json_file_name.endswith('.json'):
            continue

        output_dir = output_dir_path / json_file_name[:-5]
        output_dir.mkdir(exist_ok=True)

        tagline_list = []
        lineart_tagline_list = []
        with open('metadata/' + json_file_name, 'r', encoding='utf8') as f:
            print(f'reading metadata/{json_file_name}')
            for line in f:
                try: 
                    metadata = json.loads(line)
                    tags = metadata['tags']
                    file_id = int(metadata['id'])

                    if not is_image_exists(file_id):
                        continue

                    # use image only
                    if metadata['file_ext'] not in AVAILABLE_EXT:
                        continue

                    # use safe only
                    if metadata['rating'] is not 's':
                        continue

                    tag_id_list = list(map(lambda t: int(t['id']), tags))
                    height, width = int(metadata['image_height']), int(metadata['image_width'])
                    
                    if height == 0 or width == 0:
                        continue
                    
                    aspect = width / height

                    # drop too long or small image 
                    if height < 512 and width < 512:
                        continue
                    if not ((3/4) <= aspect <= (4/3)):
                        continue

                    # drop blacklisted tags
                    if any(tag_bl in tag_id_list for tag_bl in TAG_BLACKLIST):
                        continue

                    tagline = metadata_to_tagline(metadata)

                    # lineart는 따로 처리 (monochrome or greyscale)
                    if TAG_MONOCHROME in tag_id_list or TAG_GREYSCALE in tag_id_list:
                        move_lineart(file_id, aspect, lineart_dir)
                        lineart_tagline_list.append(tagline)
                    else:
                        move_color_image(file_id, aspect, output_dir)
                        tagline_list.append(tagline)

                    count += 1
                    if count % 1000 == 0:
                        print(f'parse count: {count}')

                except KeyError as e:
                    print(e)
                except Exception as e:
                    print(e)

                
        with (output_dir / 'tags.txt').open('w') as tag_file:
            tag_file.writelines(tagline_list)

    with (lineart_dir / 'tags.txt').open('w') as tag_file:
        tag_file.writelines(lineart_tagline_list)
