# https://fasttext.cc/docs/en/english-vectors.html
# @inproceedings{mikolov2018advances,
#   title={Advances in Pre-Training Distributed Word Representations},
#   author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
#   booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
#   year={2018}
# }

import io
import torch
import os

color_variant_files = ['body_lower.txt', 'body_upper.txt', 'body_whole.txt', 'face.txt', 'hair.txt', 'background.txt']

def load_vectors(fname, total_name_list):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')

        tag_name = tokens[0]
        if tag_name in total_name_list:
            data[tag_name] = map(float, tokens[1:])
            # print(tokens[0].encode('utf-8'))
            # print(len(tokens[1:]))
    return data


if __name__ == '__main__':
    file_name = 'wiki-news-300d-1M.vec'

    cvt_name_list = []

    extra_words = ['a', 'girl', 'boy', 'with', 'and', 'wearing']

    

    print("----CVT----")
    for fn in color_variant_files:
        with open('taglist/CVT/' + fn, 'r') as f:
            for line in f:
                tag_line = line.split()
                tag_id, tag_name = int(tag_line[0]), tag_line[1]
                tag_names = tag_name.split('_')

                for t in tag_names:
                    if t in cvt_name_list:
                        pass
                    else:
                        cvt_name_list.append(t)
    
    total_name_list = cvt_name_list + extra_words

    print(len(total_name_list))

    data = load_vectors(file_name, total_name_list)

    print(len(data))

    data_keys = list(data.keys())

    not_included_words = set(total_name_list) - set(data_keys)

    print(not_included_words)



    torch.save({
        'word_names' : data_keys,
        'word_vectors' : data
        }, os.path.join('./', 'w2v.pkl'))
