import pickle

color_invariant_files = ['body_lower.txt', 'body_upper.txt', 'body_whole.txt', 'face.txt', 'hair.txt', 'object.txt']
color_variant_files = ['body_lower.txt', 'body_upper.txt', 'body_whole.txt', 'face.txt', 'hair.txt', 'background.txt']
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
    iv_part_list = list()
    cv_part_list = list()

    tag_ids = set(tag_dict.values())

    for fn in color_invariant_files:
        with open('CIT/' + fn, 'r') as f:
            part_name = fn[:-4]
            part_list = []
            for line in f:
                tag_line = line.split()
                tag_id, tag_name = tag_line[0], tag_line[1]
                if tag_id in tag_ids:
                    iv_tag_list.add(tag_id)
                    part_list.append(tag_id)
            iv_part_list.append((part_name, part_list))

    for fn in color_variant_files:
        with open('CVT/' + fn, 'r') as f:
            part_name = fn[:-4]
            print(part_name)
            part_list = []
            for line in f:
                tag_line = line.split()
                tag_id, tag_name = tag_line[0], tag_line[1]
                if tag_id in tag_ids:
                    cv_tag_list.add(tag_id)
                    part_list.append(tag_id)
            cv_part_list.append((part_name, part_list))

    iv_tag_list = list(iv_tag_list)
    cv_tag_list = list(cv_tag_list)

    result = {
        'iv_tag_list': iv_tag_list,
        'iv_part_list' : iv_part_list,
        'cv_tag_list': cv_tag_list,
        'cv_part_list' : cv_part_list,
        'tag_dict': tag_dict,
        'tag_count_dict': tag_count_dict
    }
    with open('tag_dump.pkl', 'wb') as fw:
        pickle.dump(result, fw)

    print(f'invariant len: {len(iv_tag_list)} / cv len: {len(cv_tag_list)} / sum: {len(iv_tag_list + cv_tag_list)}')