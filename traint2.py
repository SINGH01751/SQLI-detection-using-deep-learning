import pandas as pd
from nltk.corpus import stopwords

df = pd.read_csv("C:\\proj\\sqliv2.csv", encoding='utf-16')
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
# fitting the  data to sentence column of dataset and then converting sparse matrix to array
posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()
transformed_posts = pd.DataFrame(posts)
df = pd.concat([df, transformed_posts], axis=1)
# assigning data frame column 3 to x and label column to y
X = df[df.columns[2:]]
y = df['Label']
from sklearn.model_selection import train_test_split

# splitting the preprocessed dataset in 20:80
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]
# Number of features
model = Sequential()
model.add(layers.Dense(20, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(10, activation='tanh'))
model.add(layers.Dense(1024, activation='relu'))

model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
classifier_nn = model.fit(X_train, y_train,
                          epochs=10,
                          verbose=True,
                          validation_data=(X_test, y_test),  # doing validation of training data and test data
                          batch_size=15)
pred = model.predict(X_test)  # bringing predictions from the model
for i in range(len(pred)):
    if pred[i] > 0.5:
        pred[i] = 1
    elif pred[i] <= 0.5:
        pred[i] = 0
print(accuracy_score(y_test, pred))

import pickle

model.save('my_model_cnn.h5')
with open('vectorizer_cnn', 'wb') as fin:
    pickle.dump(vectorizer, fin)
