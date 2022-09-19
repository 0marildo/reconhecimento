import cv2
from random import randrange

#Coloca alguma pre-treinada informação em faces frontais do OpenCv ( haar Cascata algoritmo)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Colocar a imagem da Cam na string
img = cv2.imread("Teste.jpg")

#Converte em cinza
cinza_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Colocar imagem cinza dentro dos parenteses 
# formata o quadrato para estar na escala do rosto
coordenadas_face = trained_face_data.detectMultiScale(cinza_img)

#Cria um retangulo em volta da face 
for (x, y, w, h) in coordenadas_face:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)

print(coordenadas_face)

cv2.imshow("Reconhecimento facial muito foda", img)
cv2.waitKey()

print("Feito")
