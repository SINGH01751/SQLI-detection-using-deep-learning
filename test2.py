import pickle

import tensorflow as tf

mymodel = tf.keras.models.load_model('my_model_cnn.h5')
myvectorizer = pickle.load(open("vectorizer_cnn", 'rb'))


def predict_attack():
    repeat = True
    beautify = ''
    for i in range(20):
        beautify += "*"
    print(beautify)
    input_val = input("Enter login info:")

    if input_val == '0':
        repeat = False

    input_val = [input_val]
    input_val = myvectorizer.transform(input_val).toarray()
    result = mymodel.predict(input_val)
    if repeat:
        if result >= 0.5:
            print("Alert! SQL Injection Detected")
        elif result < 0.5:
            print("Normal User")
            print(beautify)
            predict_attack()
    elif not repeat:
        print("Closing Detection")


predict_attack()
