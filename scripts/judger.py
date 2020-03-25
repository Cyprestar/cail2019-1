import sys
import json


def get_score(ground_truth_path, output_path):
    cnt = 0
    f1 = open(ground_truth_path, "r", encoding='utf-8')
    f2 = open(output_path, "r", encoding='utf-8')

    correct = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0

    for line in f1:
        cnt += 1
        true_data = json.loads(line)
        real_label = f2.readline().strip()
        predict_label = true_data['label']
        if predict_label == real_label:
            correct += 1
            if real_label:
                true_positive += 1
        elif predict_label:
            false_positive += 1
        else:
            false_negative += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    return 1.0 * correct / cnt, f1


if __name__ == "__main__":
    ground_truth_path = "../data/test/test.json"
    output_path = "../data/test/output.txt"
    if len(sys.argv) == 3:
        ground_truth_path = sys.argv[1]
        output_path = sys.argv[2]

    accuracy, f1 = get_score(ground_truth_path, output_path)
    print('Accuracy: {:.7f}, F1: {:.7f}.'.format(accuracy, f1))
