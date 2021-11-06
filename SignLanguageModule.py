#!/usr/bin/env python
# coding: utf-8

# # 1. Import and Install Dependencies/Libraries

# In[2]:


#### NOTE:
###### This notebook has to be converted into a python (.py) file
###### when imported in other notebooks in the same folder
###### in order for its functions to be accessible by the other notebooks


# In[ ]:


get_ipython().system('pip install tensorflow==2.5.0 opencv-python mediapipe sklearn matplotlib ')


# In[ ]:


get_ipython().system('pip install pyttsx3 # speech')


# In[1]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import string

# Sound
import pyttsx3


# # 2. Setup Folders for Data Collection

# In[ ]:


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Signs to detect
signs = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                   'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                   'hello', 'my', 'name', 'is', 'nice', 'you', 'bye', 'thank you', 'to meet'])

# Thirty videos worth of data
no_sequences = 60 #30

# Each video has 30 frames in length
sequence_length = 30

# Folder start
start_folder = 60 #30


# In[ ]:


# Create the necessary folders for the signs defined previously
for sign in signs:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, sign, str(sequence)))
        except:
            pass


# # 3. Keypoints using MP Holistic

# In[ ]:


# Downloading and leveraging the model
mp_holistic = mp.solutions.holistic

# For easier drawing of keypoints on face
mp_drawing = mp.solutions.drawing_utils 


# In[ ]:


# Function to make detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# In[ ]:


# Concatenate extracted keypoints values into numpy arrays
# If no detection is made on any of the following: pose/face/left hand/right hand, arrays are filled with zeros
# calculation of arrays zero: mediapipe landmarks * number of coordinates values (eg: x, y, z, visibility)
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# In[ ]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


# In[ ]:


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


# # 4. Speech Conversion for Real Time Detection

# In[ ]:


# Get factory reference to pyttsx3.Engine instance
engine = pyttsx3.init()
# Get current value of voices for the engine property
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[0].id)
# Change speech rate with 150 words per minute 
engine.setProperty('rate', 150)

# Function to output voice
def speak(str):
    engine.say(str)
    engine.runAndWait()
    engine.stop()


# # 5. Collecting Keypoints Values for Training/Testing Data Collection

# In[ ]:


# This function is invoked when collecting keypoints values for signs' sequences (videos)
# during training/testing data collection
def collect_keypoints():
    cap = cv2.VideoCapture(0)
    
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through signs
        for sign in signs: 
            # Loop through sequences (videos)
            for sequence in range(no_sequences):
                # Loop through video length (sequence length)
                for frame_num in range(sequence_length):
                    
                    # Read feed
                    ret, frame = cap.read()
                    
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)     
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(sign, sequence), (15, 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('Image Collection for Model', image)
                        cv2.waitKey(2000) # 2 seconds break interval to reposition before collecting next sequence(video)
                        
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(sign, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)       
                        # Show to screen
                        cv2.imshow('Image Collection for Model', image)
                    
                    # Export Keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, sign, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        
        cap.release()
        cv2.destroyAllWindows()


# # 6. Build the LSTM neural network

# In[ ]:


# Import necessary libraries for building LSTM neural network
from tensorflow.keras.models import Sequential # build sequential neural network
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard


# In[ ]:


# For monitoring neural network accuracy while training through the web app

# To access:
# 1. Open cmd, go to the folder (ConvertingSignLanguageToSpeech\Logs\train)
# 2. Enter and run the command "tensorboard --logdir=." to open TensorBoard from within the current folder
# 3. Copy the link given into a web browser to monitor the neural network while training
# or, 
# 1. Open Anaconda Prompt, activate the virtual environment of the project where Tensorflow is 
# (e.g: conda activate sign_language_to_speech)
# 2. Enter and run the command "tensorboard --logdir=PATH_TO_LOG_FILES"
# (e.g: C:\Users\Dylan\ConvertingSignLanguageToSpeech\Logs\train)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# In[ ]:


# Function to instantiate the model
def createModel():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662))) # change the value
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(700, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(700, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu')) 
    model.add(Dense(signs.shape[0], activation='softmax'))

    # Specify the loss for multi-class classification model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


# In[7]:


# Initialise a list of 26 alphabets
alphabets_list = list(string.ascii_lowercase)

print(alphabets_list)


# # 7. Real Time Detection

# In[2]:


# This function is invoked when performing the real-time detection of converting sign language to speech
# in the SignLanguageToSpeech notebook, after loading the trained LSTM neural network
def sign_to_speech():
    # New detection variables
    sequence = []
    sentence = []
    predictions = [] 
    threshold = 0.8 #0.8
    pTime = 0
    cap = cv2.VideoCapture(0)
    
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            cTime = time.time()
            fps = 1/ (cTime - pTime)
            pTime = cTime
            cv2.putText(image, f'FPS: {int(fps)}', (150, 70), cv2.FONT_HERSHEY_PLAIN,
                       3, (0, 255, 0), 2)
            
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(signs[np.argmax(res)])
                predictions.append(np.argmax(res))
                print(res[np.argmax(res)])

            # Viz logic
                # Check to make sure last 10 frames have exact same prediction for stability in prediction
                if np.unique(predictions[-10:])[0]==np.argmax(res):
                    if res[np.argmax(res)] >= threshold:
                        if len(sentence) > 0:
                            
                            # Check if predicted sign gestures returns an alphabet
#                             if len(signs[np.argmax(res)]) == 1: 
                            if signs[np.argmax(res)] in alphabets_list:
                                
                                # Two conditions when appending predicted sign gestures to sentence: 
                                # 1. Check if recent word in sentence is an alphabet
                                # 2. Check if recent word in sentence is not in signs array
                                # Concatenate the recent word with the predicted alphabet in sentence
#                                 if len(sentence[-1]) >= 1 or sentence[-1] not in signs:
                                if sentence[-1] in alphabets_list or sentence[-1] not in signs:
                                    speak(signs[np.argmax(res)])
                                    sentence[-1] = sentence[-1]+signs[np.argmax(res)]
                            
                            # Check if predicted sign gestures returns non-alphabets
                            else:
                                # Append it to sentence and output to speech
#                                 sentence.append(signs[np.argmax(res)])
#                                 speak(sentence[-1])
                                if signs[np.argmax(res)] != sentence[-1]:
                                    sentence.append(signs[np.argmax(res)])
                                    speak(sentence[-1])
                                
                        else:
                            sentence.append(signs[np.argmax(res)])
                            speak(sentence[-1])
                        
                
                # Take last five values to prevent rendering a giant sentence array
                if len(sentence) > 5:
                    sentence = sentence[-5:]

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('Sign Language Detection to Speech', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()           


# # 8. Testing Webcam

# In[ ]:


# This function is just to test the webcam is accessed for making detections
def test_camera():
    
    cap = cv2.VideoCapture(0) # Change the value to 1 if using external webcam
    
    while cap.isOpened():
        
        # Read feed
        ret, frame = cap.read()

        # Show to screen
        cv2.imshow('Test Webcam', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

