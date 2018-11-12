import pickle

color_invariant_files = ['bodypart.txt', 'face_hair.txt', 'fashion.txt', 
    'object.txt', 'pose.txt']

color_variant_files = ['colorpart.txt', 'background.txt']

essential_color_tag_files = ['eyes.txt', 'hair.txt']

non_essential_color_tag_files = ['body.txt', 'dress.txt', 'etc.txt', 'footwear.txt', 'item.txt', 'leg.txt', 'neckwear.txt', 'skin.txt', 'underwear.txt', 'upbody.txt']

tag_list_file = 'tags.txt'

if __name__ == '__main__':
    tag_dict = dict()
    tag_count_dict = dict()

    with open(tag_list_file, 'r') as f:
        for line in f:
            [tag_id, tag_name, tag_count] = line.split()
            tag_dict[tag_name] = int(tag_id)
            tag_count_dict[tag_id] = int(tag_count)

    iv_tag_list = set()
    cv_tag_list = set()
    essential_color_tag_list = []
    non_essential_color_tag_list = []

    tag_keys = set(tag_dict.keys())

    for fn in color_invariant_files:
        with open(fn, 'r') as f:
            for line in f:
                tag_name = line.split()[0]
                if tag_name in tag_keys:
                    iv_tag_list.add(tag_dict[tag_name])

    for fn in color_variant_files:
        with open(fn, 'r') as f:
            for line in f:
                tag_name = line.split()[0]
                if tag_name in tag_keys:
                    cv_tag_list.add(tag_dict[tag_name])

    for fn in essential_color_tag_files:
        temp = []
        with open(fn, 'r') as f:
            for line in f:
                tag_name = line.split()[0]
                if tag_name in tag_keys:
                    temp.append(tag_dict[tag_name])

        essential_color_tag_list.append(temp)

    for fn in non_essential_color_tag_files:
        temp = []
        with open(fn, 'r') as f:
            for line in f:
                tag_name = line.split()[0]
                if tag_name in tag_keys:
                    temp.append(tag_dict[tag_name])

        non_essential_color_tag_list.append(temp)


    iv_tag_list = list(iv_tag_list)
    cv_tag_list = list(cv_tag_list)
    essential_color_tag_list = essential_color_tag_list
    non_essential_color_tag_list = non_essential_color_tag_list


    result = {
        'iv_tag_list': iv_tag_list,
        'cv_tag_list': cv_tag_list,
        'essential_color_tag_list': essential_color_tag_list,
        'non_essential_color_tag_list': non_essential_color_tag_list,
        'tag_dict': tag_dict,
        'tag_count_dict': tag_count_dict
    }
    with open('tag_dump.pkl', 'wb') as fw:
        pickle.dump(result, fw)

    print(f'invariant len: {len(iv_tag_list)} / cv len: {len(cv_tag_list)} / sum: {len(iv_tag_list + cv_tag_list)}')
