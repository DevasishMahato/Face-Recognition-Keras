# Developed by - Devasish Mahato
# Institute - Indian Institute of Information Technology Pune
import cv2
import dlib
import tensorflow as tf
import numpy
from keras.preprocessing import image
# Load Image Information from Face Database
train_data = image.ImageDataGenerator()
test_data = image.ImageDataGenerator()
# Training & Test Sets
training = train_data.flow_from_directory('OurClass',target_size=(150,150))
testing = test_data.flow_from_directory('testClass',target_size=(150,150))
#import Keras
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Activation
from keras.models import Sequential

names=['Devasish','Disha']
#model
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation="softmax"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit_generator(training,steps_per_epoch=10,epochs=1)
s = model.evaluate_generator(testing,steps=10,verbose=0)
# Live stream
detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
for i in range(1000):
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    for det in dets:
        cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        global crop_img
        crop_img = img[det.left():det.right(), det.top():det.bottom()]
        crop_img = cv2.resize(crop_img,(150,150))
        crop_img = numpy.reshape(crop_img,(1,150,150,3))
    predicting = image.ImageDataGenerator()
    crop_img = predicting.flow(crop_img,batch_size=1)
    name = model.predict_generator(crop_img,steps=1).argmax()
    print(names[name])
    cv2.imshow('LiveStream', img)
cv2.destroyAllWindows()
