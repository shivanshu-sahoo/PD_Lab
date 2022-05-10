import streamlit as st
import pickle
import cv2 as cv
from tensorflow.keras.preprocessing import image 
import numpy as np
from tensorflow.keras.models import load_model
import serial
import time
from playsound import playsound
import os
print(os.getcwd())
model=load_model(r'.\model\self_trained\resnet_50.hdf5')
def app():
    global model
    # video_path=r'.\input_video.mp4'
    cap=cv.VideoCapture(0)

    #serialcomm = serial.Serial('COM7', 115200)
    #serialcomm.timeout = 1
    #msg = 'on'

    class_name={9:'safe_driving',1:'texting_right',2:'talking_on_phone_right',6:'texting_left',5:'talking_on_phone_left',0:'operating_radio',3:'drinking',8:'reaching_behind',7:'hair_and_makeup',4:'talking_with_passenger'}

    current_time=0
    start_time=0

    if not cap.isOpened(): 
        print('nai hua')

    FRAME_WINDOW = st.image([])

    while True: 
        ret,frame=cap.read()
        # img=frame.copy()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img=frame.copy()
        img2= cv.resize(frame,dsize=(224,224), interpolation = cv.INTER_CUBIC)
        #Numpy array
        np_image_data = np.asarray(img2)
        #maybe insert float convertion here - see edit remark!
        np_final = np.expand_dims(np_image_data,axis=0)
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        np_final=np_final.astype('float32')/255 - 0.5
        y=model.predict(np_final)
        y_class=np.argmax(y,axis=1)
        
        #print(class_name[y_class[0]])
        if(class_name[y_class[0]] != 'safe_driving'):
            if(current_time - start_time> 5):
                start_time=time.time()
                #serialcomm.write(msg.encode())
                playsound(r'.\buzzer.wav',block=False)
                
        current_time=time.time()
            
        
        cv.putText(img,class_name[y_class[0]],(35, 50), cv.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
        # cv.imshow('video',img)
        FRAME_WINDOW.image(img)
        if cv.waitKey(1) &0xFF == ord('q'):        
            break  
    #serialcomm.close()
    cap.release()
    cv.destroyAllWindows()
