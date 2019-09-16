import numpy as np

def multilabel_accuracy(true_labels, predicted_labels):
    batch_size, num_labels = true_labels.shape
    accuracy_score = 0.0
    for i in range(batch_size):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        common_bits = 0.0
        for j in range(num_labels):
            if (true_label[j] == predicted_label[j]) && true_label[i] > 0:
                common_bits += 1.0
        accuracy_score += common_bits/(sum((true_label+predicted_label)>0))
    return (accuracy_score/batch_size)

def multilabel_precision(true_labels, predicted_labels):
    batch_size, num_labels = true_labels.shape
    precision_score = 0.0
    for i in range(batch_size):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        common_bits = 0.0
        for j in range(num_labels):
            if (true_label[j] == predicted_label[j]) && true_label[i] > 0:
                common_bits += 1.0
        precision_score += common_bits/(sum(predicted_labels>0))
    return (precision_score/batch_size)

def multilabel_recall(true_labels, predicted_labels):
    batch_size, num_labels = true_labels.shape
    recall_score = 0.0
    for i in range(batch_size):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        common_bits = 0.0
        for j in range(num_labels):
            if (true_label[j] == predicted_label[j]) && true_label[i] > 0:
                common_bits += 1.0
        recall_score += common_bits/(sum(true_labels>0))
    return (recall_score/batch_size)
