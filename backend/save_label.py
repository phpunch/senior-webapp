import os
import glob
from collections import defaultdict
import shutil
from pprint import pprint
# label_list = [{
#     'filename': 'demo-000001',
#     'label': 181
# }]

def save_label(video_id, label_list):

    counter = defaultdict(int)

    for obj in label_list:
        if (obj["label"] == ""): 
            print("{} : no label".format(obj["filename"]))
            continue

        str_label = "{:03d}".format(int(obj["label"]))
        
        old_name = "{}.wav".format(obj["filename"]) # demo-000001.wav
        new_name = "{}-{}-{:06d}.wav".format(str_label, video_id, counter[str_label])
        
        src_path = os.path.join("audios", old_name)
        dst_path = os.path.join("database", str_label, new_name)
        print(src_path, dst_path)

        folder_path = os.path.join("database", str_label)
        if (not os.path.exists(folder_path)):
            os.mkdir(folder_path)
        shutil.copy2(src_path, dst_path)

        counter[str_label] += 1

def add_to_storage(storage_path):
    file_path_list = glob.glob("database/*/*")
    assert len(file_path_list) > 0, "file not found in database"
    for src_path in file_path_list:
        _, folder, file_name = src_path.split("/")
        dst_path = os.path.join(storage_path, folder, file_name)
        shutil.copy2(src_path, dst_path)
        print(dst_path)

if __name__ == "__main__":
    add_to_storage("/media/punch/DriveD/D_Download/audios_parliament")
    