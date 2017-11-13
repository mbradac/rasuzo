import face_recognition
import cv2
import sys
import os
import logging

THRESHOLD = 0.7

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
for path in os.listdir(database_path):
    person_folder_path = os.path.join(database_path, path)
    assert os.path.isdir(person_folder_path), \
            "Children of database_path should be folders"
    logger.info("Found person {} in database".format(person_folder_path))
    for image_path in os.listdir(person_folder_path):
        full_image_path = os.path.join(person_folder_path, image_path)
        loaded_image = face_recognition.load_image_file(full_image_path)
        face_encoding = face_recognition.face_encodings(loaded_image)[0]
        database_face_encodings.append(face_encoding)
        database_persons.append(person_folder_path)
        logger.info("\tFound image {} in database".format(full_image_path))

face_locations = []
frame_number = 0

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
        distances = list(face_recognition.face_distance(
                database_face_encodings, face_encoding))
        name, distance = max(zip(database_persons, distances), key=lambda x: x[1])
        # Draw rectangle for matched face
        (top, right, bottom, left) = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw rectangle for person's name if recognized
        if distance > THRESHOLD:
             font = cv2.FONT_HERSHEY_DUPLEX
             cv2.putText(frame, name,
                         (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    logger.info("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
