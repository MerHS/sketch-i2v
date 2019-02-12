import argparse

p = argparse.ArgumentParser()
p.add_argument("-s", '--skip', type=int, default=0)
p.add_argument('-d', '--delete', action='store_true')
args = p.parse_args()

text_dict = {
    'e': 'background.txt', 'w': 'face.txt', 'q': 'hair.txt',
    's': 'body_upper.txt', 'a': 'body_whole.txt',
    'x': 'body_lower.txt', 'z': 'object.txt'
}
if args.delete:
    for txt in text_dict.values():
        for dir in ['CIT', 'CVT']:
            with open(f'{dir}/{txt}', 'w') as fx:
                continue
skip_count = 0
with open('dataset_count.txt', 'r') as f:
    for i, line in enumerate(f):
        if skip_count < args.skip:
            skip_count += 1
            continue
        pos = input(str(i) + ' ' + line)
        if pos[0] == '3':
            continue
        dir = 'CIT' if pos[0] == '1' else 'CVT'
        txt = text_dict[pos[1]]
        with open(f'{dir}/{txt}', 'a') as fx:
            fx.write(line)
