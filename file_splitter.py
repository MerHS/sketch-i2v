import shutil
import os.path
import random
from random import randint
from pathlib import Path

IMAGE_ROOT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), '256px')
OUT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')

def random_move(img_dir_path, out_dir_path):
    train_path = out_dir_path / 'train'
    test_path = out_dir_path / 'test'
    vaild_path = out_dir_path / 'validation'

    tagfile_name = 'tags.txt'
    tag_txt_path = img_dir_path / tagfile_name
    
    paths = [str(train_path), str(test_path), str(vaild_path)]
    tag_paths = [train_path / tagfile_name, test_path / tagfile_name, vaild_path / tagfile_name]

    for p in [test_path, train_path, vaild_path]:
        if not p.exists():
            p.mkdir()
    
    file_id_list = []
    file_line_list = []
    with tag_txt_path.open('r') as f:
        for line in f:
            file_id = line.split()[0]
            if not ((img_dir_path / f'{file_id}.png').exists() and 
                (img_dir_path / f'{file_id}_sk.png').exists()):
                continue
            file_id_list.append(file_id)
            file_line_list.append(line)

    count = [0, 0, 0]
    with tag_paths[0].open('a') as trf, tag_paths[1].open('a') as tsf, tag_paths[2].open('a') as valf:
        tag_files = [trf, tsf, valf]

        for file_id, file_line in zip(file_id_list, file_line_list):
            file_path = img_dir_path / f'{file_id}.png'
            skt_path = img_dir_path / f'{file_id}_sk.png'

            r = randint(0, 10)
            n = 0
            if r == 1:
                n = 1
            elif r == 0:
                n = 2

            tag_file = tag_files[n]
            shutil.move(str(file_path), paths[n])
            shutil.move(str(skt_path), paths[n])
            tag_file.write(file_line)
            count[n] += 1

    print(f'moved - train({count[0]}) / test({count[1]}) / validation({count[2]})')
       
if __name__ == '__main__':
    import argparse

    random.seed(1987)
    p = argparse.ArgumentParser()
    p.add_argument("--image_root_dir", default=IMAGE_ROOT_DIRECTORY)
    p.add_argument("--out_dir", default=OUT_DIRECTORY)
    args = p.parse_args()

    out_dir_path = Path(args.out_dir)
    if not out_dir_path.exists():
        out_dir_path.mkdir()
        
    for image_dir in Path(args.image_root_dir).iterdir():
        print(f'moving {image_dir}')
        if image_dir.is_dir():
            random_move(image_dir, out_dir_path)
