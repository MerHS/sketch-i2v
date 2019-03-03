import io

def load_vectors(fname, cvt_list):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        print(tokens[0].encode('utf-8'))
        print(len(tokens[1:]))
    return data


file_name = 'wiki-news-300d-1M.vec'

data = load_vectors(file_name)

print(data)
