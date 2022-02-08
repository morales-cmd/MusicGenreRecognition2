import streamlit as st
import librosa
import numpy as np
import librosa.display
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pydub import AudioSegment
from tensorflow import keras
import cv2

# AudioSegment.ffmpeg = r"C:\\Users\\moorl\\Downloads\\ffmpeg-master-latest-win64-gpl-shared\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe"
def convert_mp3_to_wav(music_file):
  sound = AudioSegment.from_mp3(music_file)
  sound.export("music_file.wav",format="wav")

def extract(wav_file,t1,t2):

  wav = AudioSegment.from_wav(wav_file)
  wav = wav[1000*t1:1000*t2]
  wav.export("extracted.wav",format='wav')

def create_melspectrogram(wav_file):
  y,sr = librosa.load(wav_file,duration=30)
  topdB = 80
  melspec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), top_db=topdB, ref=np.max)
  melspec = melspec/-topdB
  cv2.imwrite('melspectrogram.png', cv2.resize(melspec, (1300,128))*255)

def predict(model):
  image = cv2.imread('melspectrogram.png',0)
  image = np.reshape(image,(1,128,1300))
  prediction = model.predict(image/255)
  prediction = prediction.reshape((10,))
  class_label = np.argmax(prediction)
  return class_label,prediction

class_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

model=keras.models.load_model('music_genre_rec_models/music_genre_rec_models')
st.write("Music Genre Recognition App") 
st.write("This is a Web App to predict genre of music")
file = st.sidebar.file_uploader("Please Upload Mp3 Audio File Here or Use Demo Of App Below using Preloaded Music",type=["mp3"])

if file is None:
  st.text("Please upload an mp3 file")
else:
  convert_mp3_to_wav(file)
  extract("music_file.wav",10,50)
  create_melspectrogram("extracted.wav")
  class_label,prediction = predict(model)
  st.write("## The Genre of Song is: "+ class_labels[class_label])