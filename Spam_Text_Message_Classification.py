## ------------------ Import modules ----------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from sklearn.metrics import precision_score, recall_score   # Model performance metrics
from sklearn.model_selection import StratifiedKFold # For cross validation train/test splits
from sklearn.preprocessing import StandardScaler # For feature normalization
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


## ----------------- Download NLP package and load NLP modules ----------------------------------------
# This can be done in the terminal!!!
#import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

## -------------------- Functions -------------------------------------------------------------------------

# Convert the word to "grundform"
def lemmetize(df):

    # CREATE new df
    data_clean= pd.DataFrame(columns=['Category', 'Message'])

    for i in range(len(df.index)):
        txt_row = df.iloc[i,1]              # col 1 has the text to be preprocessed and classified - A SENTENCE
        tokens = word_tokenize(txt_row)     # split it up, make it more clear
        word_lst = []
        for token in tokens:                # For every word in the sentence
            lemmetized_word = lemmatizer.lemmatize(token)       # Lemmetize the word
            word_lst.append(lemmetized_word)                     # append a new world list
        sentence = " ".join(word_lst)                           # create the lemmitize sentence

        add_row = pd.Series({'Category': df.iloc[i,0], 'Message': sentence}) #for each entry in the new dataframe, keep the label the same
        data_clean = data_clean.append(add_row, ignore_index= True)
    return data_clean

# MEN det tar fett lång tid - Något snabbare sätt?
# INPUT: String sentence
# OOUTPUT: list of numeric float values
def numeric_vector(df):

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")     # Download the model
    embeddings = embed(df['Message'])                       # Generate embeddings = the metod to transform from numeric vector
    float_vec = np.array(embeddings).tolist()               # Create list from np arrays
    df['num_vec'] = float_vec                               # Add lists as dataframe column
    df = df[['Category', 'num_vec']]                        # Sentence column does not have to be returned
    return df

# INPUT: features, labels and classficiation model
# CV 10 times, splitted into train and test, ML model trained and tested
# OUTPUT: evalutation meassures, s.a precision and recall
def model_eval(features, labels, model):

    precisions = np.array([])                        # Create empty arrays som håller precision och recall
    recalls = np.array([])                           # för varje CV iteration

    features_T = features.T
    # Crossvalidation
    get_folds= StratifiedKFold(n_splits= 10, random_state= 42, shuffle= True)
    for train_i, test_i in get_folds.split(features, labels):

        x_train, x_test = features_T[train_i], features_T[test_i]           # get training and test data
        y_train, y_test = labels[train_i], labels[test_i]
        x_train_T = x_train.T
        x_test_T = x_test.T

        model.fit(x_train_T, y_train)                                   # Train model
        predicts = model.predict(x_test_T)                                  # Get the predicted label
        # Add measurements to list for each crossvalidation
        precisions = np.append(precisions, precision_score(y_test, predicts, average= 'weighted'))
        recalls = np.append(recalls, recall_score(y_test, predicts, average= 'weighted'))

    return precisions, recalls

def box_plot(knn_precisions, knn_recalls):
    #visualize precisions
    plt.figure()
    plt.boxplot(knn_precisions, notch= True, labels= ['k-Nearest neighbors'])
    plt.title('PRECISIONS for the 10 folds for each of the model')
    plt.ylim([0.9,1.0])
    plt.show()

    #visualize recalls
    plt.figure()
    plt.boxplot(knn_recalls, notch= True, labels= ['k-Nearest neighbors'])
    plt.title('RECALLS for the 10 folds for each of the models')
    plt.ylim([0.75, 1.0])
    plt.show()

# ----------------------------------- MAIN ----------------------------------
## ---------------------------- Load, clean and transform data ------------------------------------------------
df = pd.read_csv(r'SPAM text message 20170820 - Data.csv')          # Import data
lemmatizer = WordNetLemmatizer()                                    # Create lemmatize object
new_df = lemmetize(df)                                              # lemmatized dataframe
num_df = numeric_vector(new_df)                                     # dataframe with numerica values instead of sentences
labels = num_df['Category']                                         # Get the label, i.e ham or spam
num_vector_df = pd.DataFrame(num_df['num_vec'].tolist())            # Get numeric df, i.e features

## --------------------------- Train algoritms and visualise -------------------------------------------------------
knn = KNeighborsClassifier()                                                        # Create KNN algoritm
knn_precisions, knn_recalls = model_eval(num_vector_df, labels, knn)                # CV and
print('kNN precisions and recalls:', '\n', knn_precisions, '\n', knn_recalls)
box_plot(knn_precisions, knn_recalls)                                               # Visulise the spread of precsion and recall


# Algorithms
# Nearest neghbour!
# Random forest --> Have to convert to some kind of vectors?
# Neural networks?

# Skapa en boxplot diagram - Lägg till de andra algoritmerna
# Gör en plot över hur precission och recall förändras med fler värden, i.d medelvärde! =)
# Skapa GitHub 
