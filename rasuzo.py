import face_recognition
import cv2
import sys
import os
import logging
from sklearn.linear_model import LogisticRegression

THRESHOLD = 0.0

logging.basicConfig(level="INFO")
logger = logging.getLogger("lab2")

assert len(sys.argv) == 3, "Usage: python3 rasuzo.py video_path database_path"
video_path = sys.argv[1]
database_path = sys.argv[2]

# Open the input movie file
input_movie = cv2.VideoCapture(video_path)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_movie.get(cv2.CAP_PROP_FPS)
logger.info("Input video width={}, height={}, fps={}, number_of_frames={}"
            .format(width, height, fps, length))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

# Load database of faces
database_face_encodings = []
database_persons = []
database_labels = []
for label, person_name in enumerate(os.listdir(database_path)):
    person_folder_path = os.path.join(database_path, person_name)
    assert os.path.isdir(person_folder_path), \
            "Children of database_path should be folders"
    database_persons.append(person_name)
    logger.info("Found person {} in database".format(person_folder_path))
    for image_path in os.listdir(person_folder_path):
        full_image_path = os.path.join(person_folder_path, image_path)
        loaded_image = face_recognition.load_image_file(full_image_path)
        face_encoding = face_recognition.face_encodings(loaded_image)[0]
        database_face_encodings.append(face_encoding)
        database_labels.append(label)
        logger.info("\tFound image {} in database".format(full_image_path))

face_locations = []
frame_number = 0

model = LogisticRegression(max_iter=2000, tol=0.0001, C=0.01**-1).fit(
        database_face_encodings, database_labels)

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret: break

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Find best match
        confidences = model.decision_function([face_encoding])[0]
        label, confidence = max(enumerate(confidences), key=lambda x: x[1])
        # Draw rectangle for matched face
        (top, right, bottom, left) = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw rectangle for person's name if recognized
        if confidence > THRESHOLD:
             font = cv2.FONT_HERSHEY_DUPLEX
             cv2.putText(frame, database_persons[label],
                         (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    logger.info("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
