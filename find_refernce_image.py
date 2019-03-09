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

    include_tags = [470575]
    hair_tags = [87788, 16867, 13200, 10953, 16442, 11429, 15425, 8388, 5403, 16581, 87676, 16580, 94007, 403081, 468534]
    eye_tags = [10959, 8526, 16578, 10960, 15654, 89189, 16750, 13199, 89368, 95405, 89228, 390186]


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

            person_tag = rgb_tag_list.intersection(include_tags)
            hair_tag = rgb_tag_list.intersection(hair_tags)
            eye_tag = rgb_tag_list.intersection(eye_tags)

            # print(person_tag, hair_tag, eye_tag)

            if not (len(hair_tag) == 1 and len(eye_tag) == 1 and len(person_tag) == 1):
                continue

            # print(rgb_tag_list.intersection(tag_list), tag_list)
            if len(rgb_tag_list.intersection(tag_list)) == len(tag_list):
                if file_id in result.keys():
                    result[file_id].append(rgb_file_id)
                else:
                    result[file_id] = [rgb_file_id]
        try:
            result_path.write(f"{file_id} {' '.join(str(x) for x in result[file_id])}")
            random.shuffle(result[file_id])
            for f_id in result[file_id]:
                file_path = f"../dataset/rgb_train/{f_id}.png"
                if Path(file_path).exists():
                    copyfile(file_path, f'./reference_images/{file_id}_{f_id}.png')
                    break
        except:
            print("no matching reference_images in rgb_train", file_id)
        

        

    result_path.close()
    print(result)


