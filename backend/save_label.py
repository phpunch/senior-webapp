import os
import glob
from collections import defaultdict, Counter
import shutil

# label_list = [{
#     'filename': 'demo-000001',
#     'label': 181
# }]

def save_label(video_id, label_list):

    counter = Counter()
    for obj in label_list:
        old_name = "{}.wav".format(obj["filename"]) # demo-000001.wav
        new_name = "{}-{}-{:06d}.wav".format(obj["label"], video_id, counter[obj["label"]])
        src_path = os.path.join("audios", old_name)
        dst_path = os.path.join("database", str(obj["label"]), new_name)
        print(src_path, dst_path)

        folder_path = os.path.join("database", str(obj["label"]))
        if (not os.path.exists(folder_path)):
            os.mkdir(folder_path)
        shutil.copy2(src_path, dst_path)


        counter[obj["label"]] += 1
    