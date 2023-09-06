import cv2
from keras.models import model_from_json
import numpy as np




import librosa
import librosa.display
import numpy as np
import pandas as pd
import pickle
import sounddevice
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import IPython.display as ipd  # To play sound in the notebook
import os
import sys
import warnings
import keras
from keras.models import Sequential, Model, model_from_json
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
fs= 44100
second =  int(input("Enter time duration in seconds: "))
record_voice = sounddevice.rec( int ( second * fs ) , samplerate = fs , channels = 2 )
sounddevice.wait()
write("File01.wav",fs,record_voice)
json_file =open('C:\\Users\\goyal\\Downloads\\VoiceToneEmotionAnalysis-main\\VoiceToneEmotionAnalysis-main\\model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("C:\\Users\\goyal\\Downloads\\VoiceToneEmotionAnalysis-main\\VoiceToneEmotionAnalysis-main\\saved_models\\Emotion_Model.h5")
print("Loaded model from disk")
 
# Keras optimiser
opt = keras.optimizers.Adam(lr=0.0001)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
newData,newSR= librosa.load("File01.wav")
#ipd.Audio("File01.wav")
#plt.figure(figsize=(15, 5))
#librosa.display.waveshow(newData, sr=newSR)
#newData, newSR = librosa.load("File01.wav"
 #                             ,duration=2.5
 #                             ,sr=44100
 #                             ,offset=0.5)

newSR = np.array(newSR)
mfccs = np.mean(librosa.feature.mfcc(y=newData, sr=newSR, n_mfcc=13),axis=0)
newdf = pd.DataFrame(data=mfccs).T
#newdf
newdf= np.expand_dims(newdf,axis=2)
print(newdf.shape)
newpred=loaded_model.predict(newdf)
filename = filename = 'C:\\Users\\goyal\\Downloads\\VoiceToneEmotionAnalysis-main\\VoiceToneEmotionAnalysis-main\\labels'
infile = open(filename,'rb')
lb = pickle.load(infile)
infile.close()

# Get the final predicted label
final = newpred.argmax(axis=1)
final = final.astype(int).flatten()
final = (lb.inverse_transform((final)))
if final == "female_surprise":
    final = "surprise"
elif final == "female_happy":
    final = "happy"
elif final == "female_neutral":
    final = "neutral"
elif final == "female_sad":
    final = "sad"
elif final == "female_angry":
    final = "angry"
elif final == "female_fear":
    final = "fear"
elif final == "female_disgust":
    final = "disgust"
elif final == "male_surprise":
    final = "surprise"
elif final == "male_happy":
    final = "happy"
elif final == "male_neutral":
    final = "neutral"
elif final == "male_sad":
    final = "sad"
elif final == "male_angry":
    final = "angry"
elif final == "male_fear":
    final = "fear"
elif final == "male_disgust":
    final = "disgust"
print("Voice is",final)






# from keras_preprocessing.image import load_img
json_file = open("C:\\Users\\goyal\\Downloads\\Face_Emotion_Recognition_Machine_Learning-main (1)\\Face_Emotion_Recognition_Machine_Learning-main\\emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("C:\\Users\\goyal\\Downloads\\Face_Emotion_Recognition_Machine_Learning-main (1)\\Face_Emotion_Recognition_Machine_Learning-main\\emotiondetector.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam=cv2.VideoCapture(0)
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
while True:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try: 
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            # print("Predicted Output:", prediction_label)
            # cv2.putText(im,prediction_label)
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        cv2.imshow("Output",im)
        cv2.waitKey(27)
    except cv2.error:
        pass

