from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog

import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import imutils
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn import metrics
from tkinter import ttk

top = tkinter.Tk()
top.title("Deep Learning in Medical Imaging Focusing on MRI") #designing top screen
top.geometry("1400x1200")

global file_name
global img_name
global accuracy
X = []
Y = []
global classifier
detected = ['No Tumor Detected','Tumor Detected']

def insertFile(): #function to upload tweeter profile
    global file_name
    file_name = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,file_name+" loaded\n")


def sub_model():
        global X
        global Y
        X.clear()
        Y.clear()
        for root, dirs, directory in os.walk(file_name+"/no"):
            for i in range(len(directory)):
                name_dir = directory[i]
                image = cv2.imread(file_name+"/no/"+name_dir,0)
                ret2,th2 = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                image = cv2.resize(image, (128,128))
                im2arr = np.array(image)
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(0)
                print(file_name+"/no/"+name_dir)

        for root, dirs, directory in os.walk(file_name+"/yes"):
            for i in range(len(directory)):
                name_dir = directory[i]
                image = cv2.imread(file_name+"/yes/"+name_dir,0)
                ret2,th2 = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                image = cv2.resize(image, (128,128))
                im2arr = np.array(image)
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(1)
                print(file_name+"/yes/"+name_dir)
                
        X = np.asarray(X)
        Y = np.asarray(Y)            
        np.save("Model/myimg_data.txt",X)
        np.save("Model/myimg_label.txt",Y)
    
def model():
    global X
    global Y
    X.clear()
    Y.clear()
    if os.path.exists('Model/myimg_data.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        sub_model()                
        X = np.asarray(X)
        Y = np.asarray(Y)            
        np.save("Model/myimg_data.txt",X)
        np.save("Model/myimg_label.txt",Y)
    print("printing shape of x:",X.shape)
    print("printing shape of y:",Y.shape)
    cv2.waitKey(0)
    text.insert(END,"Total number of images found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total number of classes : "+str(len(set(Y)))+"\n\n")
           
        
def cnn_path_exist():
    global accuracy
    global classifier
    
    YY = to_categorical(Y)

    index = np.arange(X.shape[0])
    np.random.shuffle(index)

    x_train = X[index]
    y_train = YY[index]

    with open('Model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)

    classifier.load_weights("Model/model_weights.h5")
    classifier.make_predict_function()   
    print(classifier.summary())
    file = open('Model/history.pckl', 'rb')
    data = pickle.load(file)
    file.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    text.insert(END,'\n\nCNN Model Generated. See black console to view layers of CNN\n\n')
    text.insert(END,"CNN Prediction Accuracy on Test Images : "+str(accuracy)+"\n")

def cnn_path_not_exist():
        global accuracy
        global classifier
        
        YY = to_categorical(Y)
    
        index = np.arange(X.shape[0])
        np.random.shuffle(index)
    
        x_train = X[index]
        y_train = YY[index]

        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
        classifier = Sequential() #alexnet transfer learning code here
        classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 1), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 2, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10,validation_split=0.2, shuffle=True, verbose=2)
        classifier.save_weights('Model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        file = open('Model/history.pckl', 'wb')
        pickle.dump(hist.history, file)
        file.close()
        file = open('Model/history.pckl', 'rb')
        data = pickle.load(file)
        file.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,'\n\nCNN Model Generated. See black console to view layers of CNN\n\n')
        text.insert(END,"CNN Prediction Accuracy on Test Images : "+str(accuracy)+"\n")



    
def CNN():

    if os.path.exists('Model/model.json'):
        cnn_path_exist()
    else:
       cnn_path_no_exist()


def predict():
    #global img_name
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    
    image = cv2.imread(filename,0)
    image = cv2.resize(image, (128,128))
    im2arr = np.array(image)
    im2arr = im2arr.reshape(1,128,128,1)
    XX = np.asarray(im2arr)
        
    predicts = classifier.predict(XX)
    print(predicts)
    cls = np.argmax(predicts)
    print(cls)
    image = cv2.imread(filename)
    image = cv2.resize(image, (800,500))
    cv2.putText(image, 'Model Identified as : '+detected[cls], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Model Identified as : '+detected[cls], image)
    cv2.waitKey(0)


style = ('times', 18, 'bold')
title = Label(top, text='Deep Learning in Medical Imaging Focusing on MRI')
title.config(bg='#1525b0', fg='gold')  
title.config(font=style)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

style1 = ('times', 12, 'bold')
text=Text(top,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=style1)

style1 = ('times', 12, 'bold')
upload = Button(top, text="Upload MRI Images Dataset", command=insertFile)
upload.place(x=50,y=550)
upload.config(font=style1)  

model = Button(top, text="Generate Images Train & Test Model (OSTU Features)", command=model)
model.place(x=290,y=550)
model.config(font=style1) 

cnn = Button(top, text="Generate Deep Learning CNN Model", command=CNN)
cnn.place(x=710,y=550)
cnn.config(font=style1) 

predict = Button(top, text="Predict Tumor", command=predict)
predict.place(x=440,y=600)
predict.config(font=style1)
 

top.config(bg='#f2d8a2')
top.mainloop()
