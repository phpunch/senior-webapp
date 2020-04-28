import os
import pickle
from collections import OrderedDict, defaultdict
from pprint import pprint
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
    # values = np.array([e[1] for e in lst])
    # prob_values = softmax(values)
    # prob_lst = []
    # for i in range(len(lst)):
    #     prob_lst.append((lst[i][0], prob_values[i]))
    # prob_lst.sort(key=lambda x: x[1], reverse=True)

    lst.sort(key=lambda x: x[1], reverse=True)
    return lst[:3]

def smoothing(ts):
    smooth_ts = [ts[0]]
    for i in range(1, len(ts) - 1):
        window = ts[i-1:i+2]
        smooth_ts.append(sum(window)/len(window))
    # smooth_ts.append(ts[-2])
    smooth_ts.append(ts[-1])
    return smooth_ts
    

def find_best_plda(folder_name):
    score_path = "{}/exp/scores/scores-prod-clean".format(folder_name)
    with open(score_path) as f:
        lines = [line.strip() for line in f.readlines()]

    _, current_audio_name, _ = lines[0].split(" ")
    max_score = -999

    prediction = []
    lst = []
    speaker_to_frame_scores = defaultdict(list)
    audio_name_list = []
    for line in lines:   
        
        _, audio_name, _ = line.split(" ")
        
        if (audio_name != current_audio_name):
            audio_name_list.append(current_audio_name)
            for (speaker, score) in lst:
                speaker_to_frame_scores[speaker].append(score)

            current_audio_name = audio_name
            max_score = -999
            lst = []

        speaker, audio_name, score_str = line.split(" ")
        score = float(score_str)
        lst.append((speaker, score))

    audio_name_list.append(audio_name)
    for (speaker, score) in lst:
        speaker_to_frame_scores[speaker].append(score)

    # check frame numbers are equal for all speakers
    num_frames = None
    for spk in speaker_to_frame_scores.keys():
        if (num_frames is None):
            num_frames = len(speaker_to_frame_scores[spk])
        assert num_frames == len(speaker_to_frame_scores[spk])

    # smoothing 
    for spk in speaker_to_frame_scores.keys():
        speaker_to_frame_scores[spk] = smoothing(speaker_to_frame_scores[spk])

    # put prediction back
    prediction_by_frame = defaultdict(list)
    for index in range(num_frames):
        for spk in speaker_to_frame_scores.keys():
            prediction_by_frame[index].append((spk, speaker_to_frame_scores[spk][index]))
    
    # find top score
    for index in range(num_frames):
        scores = find_top_scorer(prediction_by_frame[index])
        print(scores)
        prediction.append({
            "filename": audio_name_list[index],
            "label": scores[0][0],
            "scores": scores
        })
    print(prediction)
    prediction = sorted(prediction, key=lambda x: x["filename"])
    prediction = add_silence_labels(folder_name, prediction)
    return prediction

# def find_best_plda(folder_name):
#     score_path = "{}/exp/scores/scores-prod-clean".format(folder_name)
#     with open(score_path) as f:
#         lines = [line.strip() for line in f.readlines()]

#     _, current_audio_name, _ = lines[0].split(" ")
#     max_score = -999

#     prediction = []
#     lst = []
#     for line in lines:   
        
#         speaker, audio_name, score_str = line.split(" ")
#         score = float(score_str)
        
#         if (audio_name != current_audio_name):
#             scores = find_top_scorer(lst)
#             prediction.append({
#                 "filename": current_audio_name,
#                 "label": scores[0][0],
#                 "scores": scores
#             })
#             current_audio_name = audio_name
#             max_score = -999
#             lst = []
            
#         lst.append((speaker, score))

#     scores = find_top_scorer(lst)
#     prediction.append({
#             "filename": current_audio_name,
#             "label": scores[0][0],
#             "scores": scores
#     })

#     prediction = sorted(prediction, key=lambda x: x["filename"])
#     prediction = add_silence_labels(folder_name, prediction)
#     return prediction

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
    # pprint(new_prediction)
    # raise Exception
    return new_prediction

def save_prediction(prediction, folder_name, video_name):
    with open("{}/prediction.pkl".format(folder_name), "wb") as f:
        pickle.dump(prediction, f)

    with open("predict_logs/{}.pkl".format(video_name), "wb") as f:
        pickle.dump(prediction, f)

if __name__=="__main__":
    add_silence_labels()