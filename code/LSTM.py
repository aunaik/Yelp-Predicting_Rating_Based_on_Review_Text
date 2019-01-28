# coding: utf-8

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pandas as pd
import numpy as np
import nltk
from sklearn.utils import class_weight
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import optimizers
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Understood concept and code for LSTM from https://github.com/EmielStoelinga/CCMLWI

# Importing Dataset
df = pd.read_csv("review_min50.csv", header = 0, names=['Review','Rating'])


# Assigning class variable to Y
Y = df.iloc[:,1]


# Calculating polarity of each review
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
senti = df['Review'].apply(lambda Text: pd.Series(sid.polarity_scores(Text)['compound']))


# Getting maximum length of review
max_review_len = np.max(df['Review'].apply(len))


# Calculating class weights
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(Y), Y)))


# Initializing some constants
MAX_SEQUENCE_LENGTH = 100
VOCAB_SIZE = 2000
EMBED_SIZE = 100
BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_FILTERS = 32
NUM_WORDS = 3


# Tokenizing the review data
Review = df['Review']
tokenizer = Tokenizer(VOCAB_SIZE)
tokenizer.fit_on_texts(Review)
X = tokenizer.texts_to_sequences(Review)


# Making length of all reviews equal
X = pad_sequences(X, maxlen = max_review_len) # max_review_len


# Appending sentiment data
X = np.append(X, senti, 1)
max_review_len = max_review_len + 1


# Diving train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42)


# One hot encoding of response variable
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
dummy_Y = np_utils.to_categorical(encoded_Y)


# Dividing train and validation data
X_train, X_Validate, Y_train, Y_Validate = train_test_split(X_train, dummy_Y, test_size=0.1, random_state=42)


# Defining model architecture
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=max_review_len, trainable=True)) # max_review_len
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu"))
model.add(MaxPooling1D(pool_size=1))
model.add(LSTM(100))
model.add(Dense(5, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06),
               metrics=["accuracy"])
        


# Displaying the model architecture
model.summary()


# Training the model
# Fitting the train data
history = model.fit(X_train, Y_train,
          epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(X_Validate, Y_Validate),
          class_weight=class_weights)


# Predicting test class
predictions = model.predict_classes(X_test)+1


# Evaluation metrics
def evaluation(Y_test,predictions):
    # evaluate predictions
    accuracy = metrics.accuracy_score(Y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("RMSE:{0}".format(metrics.mean_squared_error(Y_test, predictions)))
    print("Classification Report")
    print(metrics.classification_report(Y_test, predictions))
    sumOfError = 0.0
    errorDict = {}
    samplingError = {}
    for i in range(5):
        samplingError[i] = {}
        for j in range(5):
            samplingError[i][j] = 0
    
    for i in range(len(Y_test)):
        error = (abs(Y_test[i] - predictions[i]))
        if error not in errorDict.keys():
            errorDict[error] = 0
        errorDict[error] += 1
        samplingError[int(Y_test[i])-1][int(predictions[i])-1] += 1
        sumOfError += error/Y_test[i]
    
    print('Total values : '+str(len(Y_test)))    
    print(errorDict)
    print(' \t1\t2\t3\t4\t5')
    print('1\t'+str(samplingError[0][0])+'\t'+str(samplingError[0][1])+'\t'+str(samplingError[0][2])+'\t'+str(samplingError[0][3])+'\t'+str(samplingError[0][4]))
    print('2\t'+str(samplingError[1][0])+'\t'+str(samplingError[1][1])+'\t'+str(samplingError[1][2])+'\t'+str(samplingError[1][3])+'\t'+str(samplingError[1][4]))
    print('3\t'+str(samplingError[2][0])+'\t'+str(samplingError[2][1])+'\t'+str(samplingError[2][2])+'\t'+str(samplingError[2][3])+'\t'+str(samplingError[2][4]))
    print('4\t'+str(samplingError[3][0])+'\t'+str(samplingError[3][1])+'\t'+str(samplingError[3][2])+'\t'+str(samplingError[3][3])+'\t'+str(samplingError[3][4]))
    print('5\t'+str(samplingError[4][0])+'\t'+str(samplingError[4][1])+'\t'+str(samplingError[4][2])+'\t'+str(samplingError[4][3])+'\t'+str(samplingError[4][4]))
    print(samplingError)
    print("MAPE : "+str((sumOfError/len(Y_test))*100))
    print("MAE : "+str(metrics.mean_absolute_error(Y_test, predictions)))



evaluation(Y_test.values,predictions)


# Plotting the graphs of accuracy and loss

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])    
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'Validation'], loc='best')
plt.title('Train vs Validation Accuracy')
plt.savefig('TrainAndValidation.png')