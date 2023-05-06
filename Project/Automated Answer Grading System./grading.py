import json
from clusterutils import *
from cluster import *
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os
import pickle


# Define the features dictionary used by vectorize_answer
features = {
    'word_count': 0.1,
    'sentence_count': 0.25,
    'clean_words': 0.65
}


def prepare_vector(answer):
    # Initialize a TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Join the list of strings into a single string
    answer_text = ' '.join(answer)

    # Fit and transform the answer to get its vector representation
    answer_vector = vectorizer.fit_transform([answer_text]).toarray()

    # Get the feature names of the vectorizer and create a dictionary of feature-value pairs
    feature_names = vectorizer.get_feature_names_out()
    vector_dict = dict(zip(feature_names, answer_vector.flatten()))

    return vector_dict


def vectorize_answer(answer):
    # Tokenize the answer into sentences
    #sentences = nltk.sent_tokenize(answer)
    # Calculate the word count
    word_count = len(nltk.word_tokenize(answer))
    
    # Calculate the sentence count
    sentence_count = len(answer)
    
    # Clean the answer and calculate the clean word count
    cleaned_answer = clean_text(answer)
    clean_word_count = len(nltk.word_tokenize(cleaned_answer))
    
   
    # Create the answer vector
    vector = {}
    for feature in features:
        if feature == 'word_count':
            vector[feature] = word_count
        elif feature == 'sentence_count':
            vector[feature] = sentence_count
        elif feature == 'clean_words':
            vector[feature] = clean_word_count
   
    return vector



def read_dataset(filename):
    with open(filename) as dataset_file:
        dataset = json.load(dataset_file)
    return dataset


def prepare_dataset(dataset):
    processed_dataset = dict()
    for question, data in dataset.items():
        answer_vectors = []
        for answer in data['Answers']:
            cleaned_answer = clean_text(answer)
            answer_vector = vectorize_answer(cleaned_answer)
            answer_vectors.append(answer_vector)
        processed_dataset[question] = {
            'marks': data['Marks'],
            'answer_vectors': answer_vectors
        }
    return processed_dataset


def build_models(dataset):
    models = dict()
    save_path = os.getcwd()  # assign current working directory to save_path
    for question, data in dataset.items():
        marks = data['marks']
        answer_vectors = data['answer_vectors']
        model = DissimilarVectorsKMeans(n_clusters=marks)
        model.fit(answer_vectors)
        
        # Calculate the average feature values for each cluster
        cluster_centers = model.cluster_centers_
        feature_names = list(answer_vectors[0].keys())
        avg_features = []
        for i in range(marks):
            cluster_vectors = [v for j, v in enumerate(answer_vectors) if model.labels_[j] == i]
            cluster_feature_values = {feature: [] for feature in feature_names}
            for vector in cluster_vectors:
                for feature in feature_names:
                    cluster_feature_values[feature].append(vector[feature])
            avg_features.append({feature: sum(values) / len(values) for feature, values in cluster_feature_values.items()})
        
        models[question] = {
            'model': model,
            'marks': marks,
            'avg_features': avg_features
        }
        print("-Model for '", question, "'\tPrepared.")
        # Save the model to disk
        model_path = os.path.join(save_path, "model", f"{sanitize_filename(question)}.pkl")
        model_dir = os.path.join(save_path, "model")
        # Check whether the specified path exists or not
        isExist = os.path.exists(model_dir)
        if not isExist:
           # Create a new directory because it does not exist
           os.makedirs(model_dir)
           print("The new directory is created!")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"-Model for '{question}' saved to '{model_path}'.")
               
    return models

def prepare_QA(QA_dataset):
    QA = dict()
    for question, answer in QA_dataset.items():
        answer_vectors = []
        vector = prepare_vector(answer['Answers'])
        QA[question] = {
        'answer': answer['Answers'],
        'answer_vector': vector
        # 'answer_vector': prepare_vector(answer['Answers'])
        }
    return QA

def evaluate_QA(questions, student_QA, models, processed_dataset):
    evaluation = dict()
    for question, data in student_QA.items():
        model = models[question]['model']
        sorted_centers = sort_by_marks(model.cluster_centers)
        answer = data['answer']
        if answer == "":
            marks = 0
            feedback = "Try to attempt this question"
            answer = "Not answered"
        else:
            label = model.predict([data['answer_vector']])[0]
            center = model.cluster_centers[label]
            if 'answer_vectors' in processed_dataset[question]:
                reference_vectors = [x for x in processed_dataset[question]['answer_vectors']]
                marks = get_marks(sorted_centers, center, reference_vectors, len(sorted_centers))
                #feedback = get_feedback(sorted_centers[-1], data['answer_vector'])
            else:
                marks = get_marks(sorted_centers, center)

        evaluation[question] = {
            'answer': answer,
            'marks_awarded': round(marks*len(sorted_centers),2),
            'max_marks': len(sorted_centers)#,
            #'feedback': feedback
        }
    return evaluation

def sanitize_filename(filename):
    """
    Removes unsupported characters from a string to create a valid file name.
    """
    # Remove characters that are not allowed in file names
    filename = re.sub(r'[^\w\-_. ]', '', filename)

    # Replace spaces with underscores
    filename = filename.replace(' ', '_')

    return filename
   

if __name__ == "__main__":
   
    print("Step 1: Reading Training Dataset ...")
    dataset = read_dataset("dataset.json")
    print("Step 2: Creating new dictionary containing the question, marks, and a list of vectorized answers...")
    processed_dataset = prepare_dataset(dataset)
    questions = list(processed_dataset)
    print("Step 3: Building Models...")
    models = build_models(processed_dataset)
    print("Step 4: Reading QA or Test Dataset...")
    QA_dataset = read_dataset("test_dataset.json")
    print("Step 5: Formatting QA or Test Dataset...")
    student_QA = prepare_QA(QA_dataset)
    print("Step 6: Evaluating QA or Test Dataset...")
    evaluation = evaluate_QA(questions, student_QA, models, processed_dataset )
    print("***************************************************")
    print("Results:")
    print("***************************************************")
    marks_secured = 0
    marks_allotted = 0
    
    for question, data in evaluation.items():
        marks_secured += data['marks_awarded']
        marks_allotted += data['max_marks']
        print("Question:", question)
        print("Answer:", data['answer'])
        print("Marks Awarded:", data['marks_awarded'])
        print("Max Marks:", data['max_marks'])
        #print("Feedback:", data['feedback'])
        print()
        
    print("Total Marks Secured:", round(marks_secured,2), "/", marks_allotted)
    #print("Percentage Secured:", round(100*marks_secured/marks_allotted,2)
    print("Percentage Marks Secured: {:0.2f}%.\n".format(100*marks_secured/marks_allotted))