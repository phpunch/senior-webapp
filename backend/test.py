with open("kaldi/exp/result_sorted.txt") as f:
    prediction = []
    prediction_list = [row.strip() for row in f.readlines()]
    for row in prediction_list:
        name, _, _, _, _, label, score = row.split(" ")
        print(name, label, score)
        prediction.append({
            name: name,
            label: label,
            score: score
        })