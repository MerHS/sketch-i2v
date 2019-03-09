import io, random
import torch
import os, pickle
from pathlib import Path
from shutil import copyfile

if __name__ == '__main__':

    line_art_tags = Path('line_number_tags.txt').open('r')

    tag_txt_path = Path('../dataset/tags.txt').open('r')
    all_data = []

    result = dict()

    result_path = Path('reference_images.txt').open('w')

    reference_images_path = Path('./reference_images')

    if not reference_images_path.exists():
        os.makedirs(reference_images_path) 

    for rgb_data in tag_txt_path:
        all_data.append(rgb_data)

    for line in line_art_tags:
        if line == '\n':
            continue
        tag_list = list(map(int, line.split(' ')))
        file_id = tag_list[0]
        tag_list = tag_list[1:]

        # find matching image

        for rgb_data in all_data:
            rgb_tag_list = list(map(int, rgb_data.split(' ')))

            rgb_file_id = rgb_tag_list[0]
            rgb_tag_list = set(rgb_tag_list[1:])

            # print(rgb_tag_list.intersection(tag_list), tag_list)
            if len(rgb_tag_list.intersection(tag_list)) == len(tag_list):
                if file_id in result.keys():
                    result[file_id].append(rgb_file_id)
                else:
                    result[file_id] = [rgb_file_id]

        result_path.write(f"{file_id} {' '.join(str(x) for x in result[file_id])}")
        
        for f_id in result[file_id]:
            file_path = f"../dataset/rgb_train/{f_id}.png"
            if Path(file_path).exists():
                copyfile(file_path, f'./reference_images/{file_id}_{f_id}.png')
                break
        

    result_path.close()
    print(result)

