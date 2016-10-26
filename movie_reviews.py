# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 09:20:50 2016

@author: Dave Kavanagh - R00013469
"""
import os
import numpy as np

#return all words in a file, calls clean_words function to process vocabulary
def get_words_in_file(text_file):
    my_file = open(text_file, "r")
    file_contents = my_file.read()    
    return file_contents.split()

#updates values in a dictionary
def update_vocabulary(vocabulary, word):
    if word not in vocabulary:
        vocabulary[word] = 1
    else:
        vocabulary[word] = vocabulary[word] + 1

#build main vocabulary of unique words
def build_vocab(paths):
    vocab = {}
    for path in paths:
        file_list = os.listdir(path)
        for text_file in file_list:
            words = get_words_in_file(path + text_file)
            for word in words:
                update_vocabulary(vocab, word)
    return vocab

#builds word frequencies for posiive and negative vocabularies
def build_sub_vocabs(vocab, paths):
    neg_vocab = {}
    pos_vocab = {}
    for word in vocab:
        neg_vocab[word] = 0
        pos_vocab[word] = 0
    unique_pos = build_vocab([paths[0]])
    unique_neg = build_vocab([paths[1]])
    for word in unique_pos:
        pos_vocab[word] = unique_pos[word]
    for word in unique_neg:
        neg_vocab[word] = unique_neg[word]
    return pos_vocab, neg_vocab

#used in naive bayes algorithm
def calculate_prior_probability(predicted_class_count, other_class_count):
    total = predicted_class_count + other_class_count
    prior_probability = np.log(float(predicted_class_count)/total)
    return prior_probability

#used in naive bayes algorithm    
def calculate_conditional_probability(vocab, dividend):
    probabilities = {}
    for word in vocab:
        count_w_c = vocab[word] + 1
        probabilities[word] = np.log(float(count_w_c) / dividend)
    return probabilities

#implement naive bayes to classify reviews
def classify_documents(path, predict_probabilities, other_probabilities, prior_probability, label):
    file_list = os.listdir(path)
    correct_predict = 0.0
    incorrect_predict = 0.0
    for my_file in file_list:
        predicted = 0.0
        other = 0.0
        words = get_words_in_file(path + my_file)
        for word in words:
            if word in predict_probabilities:
                predicted = predicted + predict_probabilities[word]
            if word in other_probabilities:
                other = other + other_probabilities[word]
        if prior_probability + predicted > prior_probability + other:
            correct_predict = correct_predict + 1
        else:
            incorrect_predict = incorrect_predict + 1
    print "Num", label, correct_predict
    print "Num not", label, incorrect_predict
    
    percentage = (correct_predict / (correct_predict + incorrect_predict)) * 100
    return percentage
        
def main():
    paths = ["LargeIMDB\\pos\\", "LargeIMDB\\neg\\"]
    
    print "Building vocabulary...",
    vocab = build_vocab(paths)
    pos_vocab, neg_vocab = build_sub_vocabs(vocab, paths)
    print "done!"
    
    print "Calculating probabilites...",
    num_pos_rvws = len(os.listdir(paths[0]))
    num_neg_rvws = len(os.listdir(paths[1]))
    prior_probability = calculate_prior_probability(num_pos_rvws, num_neg_rvws)
    pos_probabilities = calculate_conditional_probability(pos_vocab, sum(pos_vocab.values()) + len(vocab))
    neg_probabilities = calculate_conditional_probability(neg_vocab, sum(neg_vocab.values()) + len(vocab))
    print "done!"
    
    test_paths = ["smallTest\\pos\\", "smallTest\\neg\\"]
    positive_accuracy = classify_documents(test_paths[0], pos_probabilities, neg_probabilities, prior_probability, "Positive")
    negative_accuracy = classify_documents(test_paths[1], neg_probabilities, pos_probabilities, prior_probability, "Negative")
    average_accuracy = (float(positive_accuracy) + negative_accuracy) / 2
    print "Accuracy in predicting positive documents:", positive_accuracy, "%"
    print "Accuracy in predicting negative documents:", negative_accuracy, "%"
    print "Average accuracy:", average_accuracy, "%"
    print len(vocab)
        
main()