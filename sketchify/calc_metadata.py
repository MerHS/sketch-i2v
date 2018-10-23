import json 
import pickle
import os

class Tag():
    def __init__(self, id, name, category):
        self.id = id
        self.name = name
        self.category = category
        self.count = 0

    def add_count(self):
        self.count += 1

if __name__ == '__main__':
    key_set = set()
    ext_set = set()
    tag_set = dict()
    line_count = 0

    for json_file_name in os.listdir('./metadata'):
        if not json_file_name.endswith('.json'):
            continue

        
        file_line_count = 0
        with open('metadata/' + json_file_name, 'r', encoding='utf8') as f:
            for line in f:
                data = json.loads(line)
                
                key_set.update(data.keys())
                ext_set.add(data['file_ext'])

                # safe rating만 고려
                if data['rating'] is not 's':
                    continue

                file_line_count += 1

                for tag in data['tags']:
                    id = int(tag['id'])
                    
                    if id not in tag_set:
                        tag_set[id] = Tag(id, tag['name'], tag['category'])
                        tag_set[id].add_count()
                    else:
                        tag_set[id].add_count()
        
        print('metadata/' + json_file_name + ": " + str(file_line_count))

        line_count += file_line_count
                

    with open('data.pickle', 'wb') as f:
        pickle.dump({'key_set': key_set, 'ext_set': ext_set, 'tag_set': tag_set}, f)

    print("finished! count: " + str(line_count))


# key_set = {'score', 'last_commented_at', 'image_height', 'is_note_locked', 'is_pending', 
# 'pixiv_id', 'source', 'down_score', 'pools', 'parent_id', 'md5', 'up_score', 'approver_id', 
# 'created_at', 'is_rating_locked', 'rating', 'favs', 'has_children', 'last_noted_at', 'updated_at', 
# 'is_deleted', 'is_status_locked', 'image_width', 'file_size', 'is_flagged', 'is_banned', 'id', 
# 'file_ext', 'uploader_id', 'tags'}

# ext_set = {'jpeg', 'jpg', 'rar', 'swf', 'bmp', 'zip', 'png', 'webm', 'gif', 'mp4'}

# category = { 0: Tags, 1: Artists, 3: Copyrights, 4: Characters, 5: Meta }
# total count: 2262211