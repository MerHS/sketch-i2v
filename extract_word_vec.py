# https://fasttext.cc/docs/en/english-vectors.html
# @inproceedings{mikolov2018advances,
#   title={Advances in Pre-Training Distributed Word Representations},
#   author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
#   booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
#   year={2018}
# }

import io
import torch
import os, pickle
from pathlib import Path

color_variant_files = ['body_lower.txt', 'body_upper.txt', 'body_whole.txt', 'face.txt', 'hair.txt', 'background.txt']

TAG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'taglist', 'tag_dump.pkl')

def split_cvt_to_words(tag_name):
    
    temp1 = tag_name.split('_')

    temp2 = []

    if '-' in temp1[0]:
        temp2 = temp1[0].split('-')

    return temp2 + temp1

def generate_NL_sentence(this_image_tags, sections):

    section_1 = []
    section_2 = []
    section_3 = []
    
    for tag_name in this_image_tags:
        if tag_name in sections['section_1']: # hair, face
            section_1.append(tag_name)
        elif tag_name in sections['section_2']: # body
            section_2.append(tag_name)
        elif tag_name in sections['section_3']: # background
            section_3.append(tag_name)

    result = ['a', 'girl']

    if len(section_1) > 0:
        result.append('with')
        for t in section_1:
            result += split_cvt_to_words(t)
            result.append('and')
        result = result[:-1]

    if len(section_2) > 0:
        result.append('wearing')
        for t in section_2:
            result += split_cvt_to_words(t)
            result.append('and')
        result = result[:-1]

    if len(section_3) > 0:
        result.append('in')
        for t in section_2:
            result += split_cvt_to_words(t)
            result.append('and')
        result = result[:-1]

    # a girl with (green hair) and (green eyes) wearing (~~~) and (~~~) in ~~ background

    return result

def load_vectors(fname, total_name_list):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    tag_dict = {}
    word_embeddings = []
    i = 1

    for line in fin:
        tokens = line.rstrip().split(' ')

        tag_name = tokens[0]
        if tag_name in total_name_list:
            tag_dict[tag_name] = i
            word_embeddings.append(list(map(float, tokens[1:])))
            i += 1
            # print(tokens[0].encode('utf-8'))
            # print(len(tokens[1:]))
    return tag_dict, word_embeddings

def get_tag_id(tag_dump_path):
    cv_dict = dict()
    iv_dict = dict()
    id_to_name = dict()
    try:
        f = open(tag_dump_path, 'rb')
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        cv_tag_list = pkl['cv_tag_list']
        name_to_id =  pkl['tag_dict']
        id_to_count = pkl['tag_count_dict']

    except EnvironmentError:
        raise Exception(f'{tag_dump_path} does not exist. You should make tag dump file using taglist/tag_indexer.py')

    # print(name_to_id, cv_tag_list) # 'in_tree' = 213242, [123,234,23434,3244234]
    return name_to_id, cv_tag_list

def make_dataset(data_dir_path, cvt_name_list, data_size=0):

    name_to_id, cv_tag_list = get_tag_id(TAG_FILE_PATH)
    
    cv_dict_name_to_id = dict()

    for tag_name, tag_id in name_to_id.items():
        if tag_id in cv_tag_list:
            cv_dict_name_to_id[tag_name] = tag_id

    print('reading tagline')
    rgb_train_path = data_dir_path / "rgb_train"
    rgb_test_path = data_dir_path / "benchmark"
    sketch_dir_path_list = ["keras_train", "simpl_train", "xdog_train"]
    sketch_dir_path_list = list(map(lambda x : data_dir_path / x, sketch_dir_path_list))
    sketch_train_path = data_dir_path / "keras_test"
    tag_path = data_dir_path / "tags.txt"


    train_file_id_to_sentence = read_tagline_txt(
        tag_path, rgb_train_path, cv_dict_name_to_id, data_size=data_size)

    test_file_id_to_sentence = read_tagline_txt(
        tag_path, rgb_test_path, cv_dict_name_to_id, data_size=100)

    print(train_file_id_to_sentence, test_file_id_to_sentence)
    return train_file_id_to_sentence, test_file_id_to_sentence
    # we have to change word vector index from NL sentence

def read_tagline_txt(tag_txt_path, img_dir_path, cv_dict_name_to_id, data_size=0, is_train=True):


    cv_class_len = len(cv_dict_name_to_id)

    cv_dict_id_to_name = {y:x for x,y in cv_dict_name_to_id.items()}

    print("read_tagline_txt! We will use %d CVT tags" % (cv_class_len))
    # tag one-hot encoding + 파일 있는지 확인
    if not tag_txt_path.exists():
        raise Exception(f'tag list text file "{tag_txt_path}" does not exist.')

    
    cv_tag_set = set(cv_dict_name_to_id.values()) # 1423123, 213452,///

    file_id_to_sentence = dict()

    data_limited = data_size != 0
    count = 0
    count_all = 0
    all_tag_num = 0
    awful_tag_num = 0
    cv_tag_num = 0

    include_tags = [470575, 540830]
    hair_tags = [87788, 16867, 13200, 10953, 16442, 11429, 15425, 8388, 5403, 16581, 87676, 16580, 94007, 403081, 468534]
    eye_tags = [10959, 8526, 16578, 10960, 15654, 89189, 16750, 13199, 89368, 95405, 89228, 390186]


    sections = make_sections()


    # print(sections)

    with tag_txt_path.open('r') as f:
        for line in f:
            count_all += 1
            tag_list = list(map(int, line.split(' ')))
            file_id = tag_list[0]
            tag_list = set(tag_list[1:])


            # if not (img_dir_path / f'{file_id}.png').exists():
            #     continue

            # print(file_id, tag_list)
            
            # For face images
            # if len(tag_list) < 8:
            #     continue

            # one girl or one boy / one hair and eye color
            person_tag = tag_list.intersection(include_tags)
            hair_tag = tag_list.intersection(hair_tags)
            eye_tag = tag_list.intersection(eye_tags)

            if not (len(hair_tag) == 1 and len(eye_tag) == 1 and len(person_tag) == 1):
                # print(file_id, hair_tag, eye_tag)
                awful_tag_num += 1
                continue
            
            this_image_tags =[]
            tag_exist = False

            for tag in tag_list:
                if tag in cv_tag_set:
                    this_image_tags.append(cv_dict_id_to_name[tag])
                    tag_exist = True
                    cv_tag_num += 1
            if not tag_exist:
                continue

            # print(this_image_tags)

            sentence = generate_NL_sentence(this_image_tags, sections) # [red_eyes, blue_hair]

            # print(sentence)


            # sentence : a girl with ....
            
            file_id_to_sentence[file_id] = sentence

            all_tag_num += len(tag_list)
            count += 1
            if data_limited and count > data_size:
                break

    # print(file_id_to_sentence)
    print(f'count_all {count_all}, select_count {count}, awful_count {awful_tag_num}, all_tag_num {all_tag_num},  cv_tag_num {cv_tag_num}')
    return file_id_to_sentence

def make_sections():
    
    color_variant_files = ['body_lower.txt', 'body_upper.txt', 'body_whole.txt', 'face.txt', 'hair.txt', 'background.txt']

    section_1_files = ['face.txt', 'hair.txt']
    section_2_files = ['body_lower.txt', 'body_upper.txt', 'body_whole.txt']
    section_3_files = ['background.txt']


    sections = dict()
    sections['section_1'] = []
    sections['section_2'] = []
    sections['section_3'] = []

    # print("----CVT sections----")
    for fn in section_1_files:
        with open('taglist/CVT/' + fn, 'r') as f:
            for line in f:
                tag_line = line.split()
                tag_id, tag_name = int(tag_line[0]), tag_line[1]
                sections['section_1'].append(tag_name)

    for fn in section_2_files:
        with open('taglist/CVT/' + fn, 'r') as f:
            for line in f:
                tag_line = line.split()
                tag_id, tag_name = int(tag_line[0]), tag_line[1]
                sections['section_2'].append(tag_name)

    for fn in section_3_files:
        with open('taglist/CVT/' + fn, 'r') as f:
            for line in f:
                tag_line = line.split()
                tag_id, tag_name = int(tag_line[0]), tag_line[1]
                sections['section_3'].append(tag_name)                

    return sections

if __name__ == '__main__':
    file_name = 'wiki-news-300d-1M.vec'

    word_list = []

    extra_words = ['a', 'girl', 'boy', 'with', 'and', 'wearing', 'in']

    

    # print("----CVT----")
    for fn in color_variant_files:
        with open('taglist/CVT/' + fn, 'r') as f:
            for line in f:
                tag_line = line.split()
                tag_id, tag_name = int(tag_line[0]), tag_line[1]
                tag_names = tag_name.split('_')

                for t in tag_names:
                    if '-' in t:
                        t_sub = t.split('-')
                        # print(t_sub)

                        for temp in t_sub:
                            if temp in word_list:
                                pass
                            else:
                                word_list.append(temp)
                    else:                        
                        if t in word_list:
                            pass
                        else:
                            word_list.append(t)
        
    total_name_list = word_list + extra_words

    # print(len(total_name_list))

    tag_dict, word_embeddings = load_vectors(file_name, total_name_list)

    print(tag_dict, word_embeddings)
    # print(len(data))


    result = {
        'tag_dict' : tag_dict,
        'word_embeddings' : word_embeddings
    }

    with open('w2v.pkl', 'wb') as fw:
        pickle.dump(result, fw)

    data_dir_path = Path('../dataset/')
    train_file_id_to_sentence, test_file_id_to_sentence = make_dataset(data_dir_path, word_list)

    result = {
        'train' : train_file_id_to_sentence,
        'val' : test_file_id_to_sentence
        }

    with open('tag2sentence.pkl', 'wb') as fw:
        pickle.dump(result, fw)


