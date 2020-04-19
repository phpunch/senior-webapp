import os
import pickle
from collections import OrderedDict
import numpy as np

def add_silence_labels(folder_name, prediction): # if no-label was not added, it might be an error at frontend!
    audio_name = sorted([i[:-4] for i in os.listdir("{}/audios".format(folder_name))])
    i = 0
    j = 0
    complete_prediction = []
    while (i < len(audio_name) and j < len(prediction)):

        if (prediction[j]["filename"] != audio_name[i]):
            complete_prediction.append({
                "filename": audio_name[i],
                "label": "",
                "scores": [("", 0)]
            })
            i += 1
        else:
            complete_prediction.append(prediction[j])
            i += 1; j += 1
    while (i < len(audio_name)):
        complete_prediction.append({
                "filename": audio_name[i],
                "label": "",
                "scores": [("", 0)]
        })
        i += 1
    return complete_prediction

def softmax(numbers):
    expo = np.exp(numbers)
    sum_exponentials = sum(expo)
    result = expo/sum_exponentials
    return result

def find_top_scorer(lst):
    # print(len(lst))
    values = np.array([e[1] for e in lst])
    # print(values)
    # print(softmax(values))
    prob_values = softmax(values)

    prob_lst = []
    for i in range(len(lst)):
        prob_lst.append((lst[i][0], prob_values[i]))
    prob_lst.sort(key=lambda x: x[1], reverse=True)
    # print(prob_lst)
    # raise Exception
    return prob_lst[:3]

def find_best_plda(folder_name):
    score_path = "{}/exp/scores/scores-prod-clean".format(folder_name)
    with open(score_path) as f:
        lines = [line.strip() for line in f.readlines()]


    

    current_audio_name = "demo-000000"
    max_score = -999

    prediction = []
    lst = []
    for line in lines:   
        
        speaker, audio_name, score_str = line.split(" ")
        score = float(score_str)
        
        if (audio_name != current_audio_name):
            scores = find_top_scorer(lst)
            prediction.append({
                "filename": current_audio_name,
                "label": scores[0][0],
                "scores": scores
            })
            current_audio_name = audio_name
            max_score = -999
            lst = []
            
        lst.append((speaker, score))

    scores = find_top_scorer(lst)
    prediction.append({
            "filename": current_audio_name,
            "label": scores[0][0],
            "scores": scores
        })

    prediction = sorted(prediction, key=lambda x: x["filename"])
    prediction = add_silence_labels(folder_name, prediction)

    with open("{}/prediction.pkl".format(folder_name), "wb") as f:
        pickle.dump(prediction, f)

if __name__=="__main__":
    add_silence_labels()