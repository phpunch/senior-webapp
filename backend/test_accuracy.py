import os
import pickle
from pprint import pprint

def open_file(file):
    with open(file, "rb") as f:
        label_logs = pickle.load(f)
        return label_logs

def test_accuracy(truth, predict):
    num = len(truth)
    correct = 0
    for i in range(len(truth)):
        print(truth[i], predict[i])
        if (truth[i] == predict[i]):
            print("Y")
            correct += 1
    return correct/num

if __name__ == "__main__":
    label_log = open_file('label_logs/kHk5muJUwuw.pkl')
    predict_log = open_file('predict_logs/kHk5muJUwuw.pkl')
    truth = []
    predict = []
    for obj in label_log['label_list']:
        truth.append(obj["label"])
    for obj in predict_log:
        predict.append(obj["label"])
    acc = test_accuracy(truth, predict)
    print(acc)