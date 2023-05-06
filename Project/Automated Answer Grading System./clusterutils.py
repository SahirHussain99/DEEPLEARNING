import math
import string
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()


# def calculate_weight(vector):
#     feature_weights = {
#     'word_count': 0.2,
#     'sentence_count': 0.2,
#     'correct_ratio': 0.1,
#     'clean_words': 0.5
#     }
#     vector_weight = 0
#     for feature, weight in feature_weights.items():
#         vector_weight += weight * vector[feature]
#     return vector_weight

# def calculate_weight(vector):
#     feature_weights = {
#         'word_count': 0.2,
#         'sentence_count': 0.2,
#         'correct_ratio': 0.1,
#         'clean_words': 0.5
#     }
#     vector_weight = 0
#     for feature, weight in feature_weights.items():
#         if feature == 'clean_words':
#             # Use the logarithm of the clean word count for this feature
#             vector_weight += weight * math.log(vector[features.index(feature)] + 1)
#         else:
#             vector_weight += weight * vector[features.index(feature)]
#     return vector_weight


def sort_by_marks(cluster_centers):
    # Calculate the weights of each cluster center
    weights = [calculate_weight(center) for center in cluster_centers]
    
    #print(f"weights : {weights}")
    
    # Sort the cluster centers by weight in descending order
    sorted_centers = [center for _, center in sorted(zip(weights, cluster_centers), reverse=True)]
    
    return sorted_centers


# def sort_by_marks(cluster_centers):
#     cluster_centers = list(cluster_centers)
#     cluster_centers.sort(key=calculate_weight)
#     return cluster_centers


def get_marks(cluster_centers, matching_centroid, reference_vectors, max_marks):
    # Calculate the ratios of the features to the reference answers' features
    feature_ratios = {}
    for feature in matching_centroid:
        feature_sum = sum(vec[feature] for vec in reference_vectors)
        if feature_sum > 0:
            feature_ratios[feature] = matching_centroid[feature] / feature_sum
        else:
            feature_ratios[feature] = 0

    # Calculate the weighted average of the feature ratios
    weights = {'word_count': 0.2, 'sentence_count': 0.2, 'clean_words': 0.5}
    weighted_sum = sum(feature_ratios[feature] * weights[feature] for feature in weights)
    marks = weighted_sum * max_marks

    return round(marks, 2)

def get_feedback(best_answer, student_answer):
    feedback = []
    if best_answer['clean_words'] > student_answer['clean_words']:
        feedback.append("Increase content.")
    if best_answer['sentence_count'] > student_answer['sentence_count']:
        feedback.append("Improve presentation.")
    # if best_answer['correct_ratio'] > student_answer['correct_ratio']:
    #     feedback.append("Reduce spelling mistakes.")
    if feedback == []:
        feedback.append("Proper answer.")
    return "\n".join(feedback)


def calculate_weight(vector):
    vector_weight = 0
    feature_weights = {
    'word_count': 0.2,
    'sentence_count': 0.2,
    'correct_ratio': 0.1,
    'clean_words': 0.5
    }
    for feature, weight in feature_weights.items():
        if feature in vector:
            vector_weight += weight * vector[feature]
    return vector_weight
    
#     vector_weight = 0
#     for feature, weight in feature_weights.items():
#         vector_weight += weight * vector[feature]
#     return vector_weight

def sort_by_marks(cluster_centers):
    # Calculate the weights of each cluster center
    weights = [calculate_weight(center) for center in cluster_centers]
    
    # Sort the cluster centers by weight in descending order
    sorted_centers = [center for _, center in sorted(zip(weights, cluster_centers), reverse=True)]
    
    return sorted_centers


def get_correct_ratio(reference_answer, answer):
    """
    Returns the ratio of words in the student's answer that are present in the
    reference answer.
    """
    reference_set = set(word_tokenize(reference_answer.lower()))
    answer_set = set(word_tokenize(answer.lower()))
    common_set = reference_set.intersection(answer_set)
    return len(common_set) / len(answer_set)

def clean_text(text):
    # Remove punctuation and lowercase the text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Convert the list of tokens back to a string
    cleaned_text = " ".join(words)

    return cleaned_text

def clean_answer(answer):
    """
    Cleans and lemmatizes the answer.
    """
    words = word_tokenize(answer)#, reference_answer)
    word_count = len(words)
    sentence_count = len(nltk.sent_tokenize(answer))
    #correct_ratio = get_correct_ratio(reference_answer, answer)
    clean_words = len(set(words))
    
    # Remove stop words, lemmatize the words, and join them back into a string
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    cleaned_answer = ' '.join(words)
    
    return {
        'answer': cleaned_answer,
        'word_count': word_count,
        'sentence_count': sentence_count,
        #'correct_ratio': correct_ratio,
        'clean_words': clean_words
    }



