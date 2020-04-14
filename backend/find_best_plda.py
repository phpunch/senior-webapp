import os
import pickle
from collections import OrderedDict, defaultdict
from pprint import pprint

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



def find_best_plda(folder_name):
    score_path = "{}/exp/scores/scores-prod-clean".format(folder_name)
    with open(score_path) as f:
        lines = [line.strip() for line in f.readlines()]


    def find_top_scorer(lst):
        # print(len(lst))
        lst.sort(key=lambda x: x[1], reverse=True)
        return lst[:5]

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
    return prediction

def post_process(prediction):
    temp_prediction = prediction
    prediction_dict_list = []
    for index in range(len(temp_prediction) - 1):
        right_predict = temp_prediction[index + 1]["scores"]
        current_predict = temp_prediction[index]["scores"]
        
        new_current_scores = defaultdict(float)
        label_id, score = right_predict[0]
        for current_label_id, current_score in current_predict:
            if (current_label_id == label_id):
                current_score += 8
            new_current_scores[current_label_id] = current_score
        prediction_dict_list.append(new_current_scores)

    two_pass_prediction_dict_list = [prediction_dict_list[0]]
    for index in range(1, len(temp_prediction)):
        left_predict = temp_prediction[index - 1]["scores"]
        current_predict = temp_prediction[index]["scores"]
        
        try:
            new_current_scores = prediction_dict_list[index]
        except:
            new_current_scores = defaultdict(float)

        label_id, score = left_predict[0]
        for current_label_id, current_score in current_predict:
            if (current_label_id == label_id):
                new_current_scores[current_label_id] += 8
            if index == len(temp_prediction) - 1:
                new_current_scores[current_label_id] = current_score
        two_pass_prediction_dict_list.append(new_current_scores)
    # for i in range(len(prediction)):
    #     pprint(prediction[i])
    #     pprint(two_pass_prediction_dict_list[i])
    #     print()
    
    new_prediction = []
    for i in range(len(two_pass_prediction_dict_list)):
        prediction_dict = dict(sorted(two_pass_prediction_dict_list[i].items(), key=lambda x: x[1],reverse=True))
        scores = [(k,v) for k,v in prediction_dict.items()][:3]
        new_prediction.append({
            "filename": prediction[i]["filename"],
            "label": scores[0][0],
            "scores": scores
        })
    pprint(new_prediction)
    # raise Exception
    return new_prediction

def save_prediction(prediction, folder_name):
    with open("{}/prediction.pkl".format(folder_name), "wb") as f:
        pickle.dump(prediction, f)

if __name__=="__main__":
    add_silence_labels()