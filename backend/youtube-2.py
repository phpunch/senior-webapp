import glob
import os
path_list = glob.glob('audios/*')
with open("wav.scp", mode='w') as f:
    for path in path_list:
        name = os.path.basename(path)[:-4]
        abs_path = os.path.abspath(path)
        # print("{0} {1}\n".format(name, abs_path))
        f.write("{0} {1}\n".format(name, abs_path))