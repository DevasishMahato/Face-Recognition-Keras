''' Developed by - Devasish Mahato
 Institute Indian Institute of Information Technology Pune '''
import cv2
import dlib
import numpy
from keras.preprocessing import image
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Activation
from keras.models import Sequential
import h5py

# Load Image Information from Face Database
data_generator = image.ImageDataGenerator()

# Training & Test Sets
training = data_generator.flow_from_directory('OurClass',target_size=(150,150))
testing = data_generator.flow_from_directory('testClass',target_size=(150,150))

 # Give String name to class indices
names=['Devasish','Disha']

#########################################
#                                       #
#    Colutional Neural Netwrork Model   #
#                                       #
#########################################

def conv2D_model(input_shape_value):
    
    model = Sequential()
    
    model.add(Conv2D(32,(3,3),input_shape = input_shape_value))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(16,(3,3),input_shape = input_shape_value))
    model.add(Activation("leakyrelu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(2,activation="softmax"))

    metric = 'accuracy'
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=[metric])
    model.fit_generator(training,steps_per_epoch=10,epochs=4)
    
    s = model.evaluate_generator(testing,steps=10,verbose=0)
    
    return(model)
    
model = conv2D_model((150,150,3))
model.save('face_trained.h5')

# Create the face detector object.
detector = dlib.get_frontal_face_detector()

# Capture one frame to get its size.
cam = cv2.VideoCapture(0)

#################################################
#                                               #
#   Classifying in frames and Object Tracking   #
#                                               #
#################################################
color_green = (0,255,0)
line_width = 3

for i in range(10):
    ret_val, img = cam.read()
    #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    predicting = image.ImageDataGenerator()
    tracker = cv2.TrackerBoosting_create()
    dets = detector(img)
    for det in dets:
        
        # Draw rectangle around Face with color:Green and line width: 3pt
        cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        
        # Crop Face from the whole image frame
        crop_img = img[det.left():det.right(), det.top():det.bottom()]
        
        # Create boundary box as(x, y, width, height) instead of four corner points
        bbox = (det.left(), det.top(), det.right() - det.left() ,det.bottom() - det.top())
        
        # Resize the cropped image to (150,150) which is our input shape for all the training as well as valiadtion data
        crop_img = cv2.resize(crop_img,(150,150))
        
        # Reshape in the cropped image into single rowed numpy array which contains (150 x 150) points with 3 color-values(BGR)
        crop_img = numpy.reshape(crop_img,(1,150,150,3))
        
        # Predicting the class of the cropped image
        crop_img = predicting.flow(crop_img,batch_size=1)
        name = model.predict_generator(crop_img,steps=1).argmax()
        
        # Add class name in image
        cv2.putText(img, names[name], (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    # Project image in video frame
    cv2.imshow('LiveStream', img)
        
    # Add tracker for face boundaries in image
    check = tracker.init(img, bbox)
    
    for i in range(10):
        # Read a new frame
        ok, frame = cam.read()
        if not ok:
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.putText(frame, names[name], (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            break
        
        # Project image in video frame
        cv2.imshow('LiveStream', frame)

        cv2.waitKey(1)
    # Exit if ESC pressed    
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
