import cv2
from random import randrange

#Coloca alguma pre-treinada informação em faces frontais do OpenCv ( haar Cascata algoritmo)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Colocar vídeo da cam 
webcam = cv2.VideoCapture(0)

#lê todos os frames
while True:
    success_frame_read, frame = webcam.read()
    video_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Programador muito foda", frame)

    face_oordenadas = trained_face_data.detectMultiScale(video_cinza)
    for (x, y, w, h) in face_oordenadas:
        cv2.rectangle(webcam, (x,y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)))


    done = cv2.waitKey(1)
    
    #finalizar o programa, apertar Q
    if done==81 or done==113:
        break  
    
webcam.release()