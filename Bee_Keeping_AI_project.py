# %%
!pip install librosa numpy pandas tensorflow scikit-learn matplotlib

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers



# %%
# read the csv file
df=pd.read_csv("data\Bee_hive.csv")
# to remove the spaces 
df.columns = df.columns.str.strip()
print(df.columns)
df.head()




# %%

# to check the total number of audio clip available in audio folder
audio_folder = "data/audio"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

print(f"Total audio files found: {len(audio_files)}")
print(audio_files)  # prints all filenames


# %%
#  set audio parameters
SR = 22050      #Sampling rate
DURATION = 10    #duration of each audio clip in seconds
TARGET_LEN= SR*DURATION    #total samples for fixed-length audio

# %%
# load and fix audio length
def load_and_fix(audio_path):
    y,_=librosa.load(audio_path,sr=SR)    # load audio
    if len(y)<TARGET_LEN:
        y=np.pad(y,(0,TARGET_LEN-len(y))) # pad with zeros if too short
    else:
        y=y[:TARGET_LEN] #trim if too long
    return y      
audio_path = "data/audio/Queenbee_1.wav"
y_fixed = load_and_fix(audio_path)
print(len(y_fixed))  # This will always print TARGET_LEN


# %%
# Extraction Features(MFCCs)
def extract_features(y):
    # y:fixed length audio waveform
    mfccs=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=20)  # 20 MFCC Features
    return np.mean(mfccs.T,axis=0)  #average over time,result shape:(20,)


# %%
# apply feature extraction to all clips
X=[]   # features
y_health=[]   #labels for health
y_queen=[]      #labels for queen presence
y_Prod=[]      # labels for productivity
y_sound_type = []  # labels for bee sound or not
for _, row in df.iterrows():
    # file_path = f"data/audio/{row['file_name']}"
    file_path = os.path.join("data/audio", row['file_name'])
    # Load and fix audio length
    y_wave=load_and_fix(file_path)
    #extract features
    feat=extract_features(y_wave)
    X.append(feat)
   #convert labels to 0/1
    y_health.append(0 if row['Health']=="healthy" else 1)
    y_queen.append(0 if row['Queen']=="absent" else 1) 
    y_Prod.append(0 if row['Productivity']=="low" else 1)
    y_sound_type.append(1 if row['Sound_Type']=="bee" else 0)   # 1 = bee, 0 = not bee  
 # convert list to numpy arrays
X= np.array(X)
y_health=np.array(y_health)
y_queen=np.array(y_queen)
y_Prod=np.array(y_Prod)
y_sound_type=np.array(y_sound_type)
print("Feature Shape:",X.shape)
    


