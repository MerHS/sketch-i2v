import json 
import pickle
import os
from pathlib import Path

# directory path of 512px or original folder of danbooru2017 dataset
IMAGE_DIRECTORY = 'C:\\Users\\starv\\work\\dataset\\liner_test'

TAG_SIMPLE = 412368
TAG_WHITE = 515193

image_dir_path = Path(IMAGE_DIRECTORY)

if not image_dir_path.exists():
    raise Exception('Directory of image "'+ IMAGE_DIRECTORY + '" does not exists.')

def is_image_exists(file_id):
    image_path = image_dir_path / f'{file_id}_sk.png'
    return image_path.exists()

def metadata_to_tagline(metadata):
    id_list = map(lambda tag: tag['id'], metadata['tags'])
    return metadata['id'] + ' ' + ' '.join(id_list) + '\n'

if __name__ == '__main__':
    count = 0
    
    with open('C:\\Users\\starv\\work\\dataset\\tags.txt', 'a+') as fw:
        for json_file_name in os.listdir('../metadata'):
            if not json_file_name.endswith('.json'):
                continue

            with open('../metadata/' + json_file_name, 'r', encoding='utf8') as f:
                print(f'reading metadata/{json_file_name}')
                for line in f:
                    metadata = json.loads(line)
                    tags = metadata['tags']
                    file_id = metadata['id']
                    
                    if not is_image_exists(file_id):
                        continue

                    tagline = metadata_to_tagline(metadata)
                    fw.write(tagline)
                    count += 1
                    if count % 1000 == 0:
                        print(count)

    print(count)
