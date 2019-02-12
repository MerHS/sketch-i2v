def get_count(tag_tuple):
    return tag_tuple[1][1]

if __name__ == '__main__':
    tag_dict = dict()
    with open('tags.txt', 'r') as f:
        for line in f:
            [tag_id, tag_name, tag_count] = line.split()
            tag_dict[int(tag_id)] = [tag_name, 0]
    with open('dataset_tags.txt', 'r') as f:
        for line in f:
            tags = map(int, line.split()[1:])
            for tag_id in tags:
                if tag_id not in tag_dict:
                    continue
                tag_dict[tag_id][1] += 1
    tag_list = list(tag_dict.items())
    tag_list.sort(key=get_count, reverse=True)
    with open('dataset_count.txt', 'w') as f:
        for (tag_id, [tag_name, tag_count]) in tag_list:
            f.write(f'{tag_id} {tag_name} {tag_count}\n')
        