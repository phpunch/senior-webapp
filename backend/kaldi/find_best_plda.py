import os
import pickle
from collections import OrderedDict


def find_best_plda():
    score_path = "exp/scores/scores-prod-clean"
    with open(score_path) as f:
        lines = [line.strip() for line in f.readlines()]


    def find_top_scorer(lst):
        # print(len(lst))
        lst.sort(key=lambda x: x[1], reverse=True)
        return lst[:3]

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


    with open("prediction.pkl", "wb") as f:
        pickle.dump(prediction, f)