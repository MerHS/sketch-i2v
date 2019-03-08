import io, random
import torch
import os, pickle
from pathlib import Path


if __name__ == '__main__':


    f = open('test_change_color_sentence.pkl', 'rb')
    pkl = pickle.load(f)
    test_change_color_sentence = pkl['data']

    hair_tags_names = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'purple hair', 'pink hair', 'silver hair', 'green hair', 'red hair', 'white hair', 'orange hair', 'grey hair', 'aqua hair', 'lavender hair', 'light brown hair']
    eye_tags_names = ['blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'black eyes', 'aqua eyes', 'orange eyes', 'grey eyes', 'silver eyes', 'red framed eyewear', 'black framed eyewear']

    hair_tags_names_to_id = {'blonde hair' : 87788, 'brown hair' : 16867, 'black hair' : 13200, 'blue hair' : 10953, 'purple hair':  16442, 'pink hair' : 11429, 'silver hair' : 15425, 'green hair' : 8388, 'red hair' : 5403, 'white hair' : 16581, 'orange hair' : 87676, 'grey hair' : 16580, 'aqua hair' : 94007, 'lavender hair' : 403081, 'light brown hair' : 468534}

    eye_tags_names_to_id = {'blue eyes' : 10959, 'red eyes' : 8526, 'brown eyes' : 16578, 'green eyes' : 10960, 'purple eyes' : 15654, 'yellow eyes' : 89189, 'pink eyes' : 16750, 'black eyes' : 13199, 'aqua eyes' : 89368, 'orange eyes' : 95405, 'grey eyes' : 89228, 'silver eyes' : 390186, 'red framed eyewear' : 1373022, 'black framed eyewear' : 1373029}

    tag_txt_path = Path("./tags_from_test_change_color_sentence.txt")

    content = ""

    with tag_txt_path.open('w') as f:

        for file_id in test_change_color_sentence:

            tags = test_change_color_sentence[file_id][0]

            sentence = ' '.join(tags)

            hair_tag = None
            eye_tag = None

            for hair_tags_name in hair_tags_names:
                if hair_tags_name in sentence:
                    hair_tag = hair_tags_names_to_id[hair_tags_name]
                    break

            for eye_tags_name in eye_tags_names:
                if eye_tags_name in sentence:      
                    eye_tag = eye_tags_names_to_id[eye_tags_name] 
                    break    

            WB = 515193
            
            content += f"{file_id} {hair_tag} {eye_tag} {WB}\n"
        f.write(content)
        f.close()

