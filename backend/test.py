# with open("kaldi/exp/result_sorted.txt") as f:
#     prediction = []
#     prediction_list = [row.strip() for row in f.readlines()]
#     for row in prediction_list:
#         name, _, _, _, _, label, score = row.split(" ")
#         print(name, label, score)
#         prediction.append({
#             name: name,
#             label: label,
#             score: score
#         })

import pickle
from pprint import pprint
with open("kaldi/prediction.pkl", "rb") as f:
    prediction = pickle.load(f)
    lst = []
    for filename in prediction.keys():
        lst.append({
            "filename": filename,
            "scores": prediction[filename]
        })
    pprint(lst)
    # for row in prediction_list:
    #     name, _, _, _, _, label, score = row.split(" ")
    #     print(name, label, score)
    #     prediction.append({
    #         name: name,
    #         label: label,
    #         score: score
    #     })