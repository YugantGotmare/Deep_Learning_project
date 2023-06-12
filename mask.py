import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained Keras model from .h5 file
model = load_model('best_model_mask.h5', compile=False)

# Define the classes and their corresponding colors
classes = {0: 'Mask', 1: 'No Mask'}
colors = {0: (0, 255, 0), 1: (0, 0, 255)}

# Load the face detection model from OpenCV
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Loop over each face and predict if they are wearing a mask or not
    for (x, y, w, h) in faces:
        # Extract the face ROI from the grayscale image
        face = gray[y:y + h, x:x + w]
        
        # Resize the face ROI to match the input size of the model
        face = cv2.resize(face, (224, 224))
        
        # Normalize the pixel values to be between 0 and 1
        face = face.astype("float") / 255.0
        
        # Add a batch dimension to the input image and pass it through the model
        face = np.expand_dims(face, axis=0)
        preds = model.predict(face)
        
        # Get the predicted class label and color
        label = classes[np.argmax(preds)]
        color = colors[np.argmax(preds)]
        
        # Draw the bounding box and class label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    
    # Display the output frame
    cv2.imshow('Mask Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
