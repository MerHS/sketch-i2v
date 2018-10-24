import pickle

color_invariant_files = ['bodypart.txt', 'face_hair.txt', 'fashion.txt', 
    'object.txt', 'pose.txt', 'background.txt']
color_variant_file = 'colorpart.txt'
tag_list_file = 'tags.txt'

if __name__ == '__main__':
    tag_dict = dict()
    with open(tag_list_file, 'r') as f:
        for line in f:
            [tag_id, tag_name, _] = line.split()
            tag_dict[tag_name] = int(tag_id)

    iv_tag_list = list()
    cv_tag_list = list()

    tag_keys = set(tag_dict.keys())

    for fn in color_invariant_files:
        with open(fn, 'r') as f:
            for line in f:
                tag_name = line.split()[0]
                if tag_name in tag_keys:
                    iv_tag_list.append(tag_dict[tag_name])

    with open(color_variant_file, 'r') as f:
        for line in f:
            tag_name = line.split()[0]
            if tag_name in tag_keys:
                cv_tag_list.append(tag_dict[tag_name])

    result = {
        'iv_tag_list': iv_tag_list, 
        'cv_tag_list': cv_tag_list,
        'tag_dict': tag_dict
    }
    with open('tag_dump.pkl', 'wb') as fw:
        pickle.dump(result, fw)

    print(f'invariant len: {len(iv_tag_list)} / cv len: {len(cv_tag_list)} / sum: {len(iv_tag_list + cv_tag_list)}')