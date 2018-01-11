import face_recognition
import cv2
import sys
import os
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression

MAX_NUM_INTERPOLATED_FRAMES = 60
MATCH_CONFIDENCE_THRESHOLD = 0.0
INTERPOLATION_MATCH_THRESHOLD_RATIO = 0.5

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
face_recognized_movie = cv2.VideoWriter(
        "face_recognized.avi", fourcc, fps, (width, height))
deidentification_movie = cv2.VideoWriter(
        "deidentification.avi", fourcc, fps, (width, height))

preprocess_start_timepoint = datetime.now()

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
        logger.info("\tFound image {} in database".format(full_image_path))
        loaded_image = face_recognition.load_image_file(full_image_path)
        face_encoding = face_recognition.face_encodings(loaded_image)[0]
        database_face_encodings.append(face_encoding)
        database_labels.append(label)
        logger.info("\tProcessed image {}".format(full_image_path))

face_locations = []
frame_number = 0

model = LogisticRegression(max_iter=2000, tol=0.0001, C=0.01**-1).fit(
        database_face_encodings, database_labels)

preprocess_duration = datetime.now() - preprocess_start_timepoint
process_start_timepoint = datetime.now()

matches = []

# Find faces in each frame of video
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    # Quit when the input video file ends
    if not ret: break
    frame_number += 1

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    frame_matches = []
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Find best match
        confidences = model.decision_function([face_encoding])
        if confidences.ndim == 2:
            confidences = confidences[0]
        label, confidence = max(enumerate(confidences), key=lambda x: x[1])
        frame_matches.append((face_location, label
            if confidence > MATCH_CONFIDENCE_THRESHOLD else None))

    matches.append(frame_matches)
    logger.info("Analyzed frame {} / {}".format(frame_number, length))

# Interpolate matches
for i, frame_matches in enumerate(matches):
    for match in frame_matches:
        location, label = match
        top, right, bottom, left = location
        width, height = abs(right - left), abs(top - bottom)

        def future_match_close(top2, right2, bottom2, left2):
            if abs(top2 - top) / height > INTERPOLATION_MATCH_THRESHOLD_RATIO:
                return False
            if abs(bottom2 - bottom) / height > INTERPOLATION_MATCH_THRESHOLD_RATIO:
                return False
            if abs(left2 - left) / width > INTERPOLATION_MATCH_THRESHOLD_RATIO:
                return False
            if abs(right2 - right) / width > INTERPOLATION_MATCH_THRESHOLD_RATIO:
                return False
            return True

        def find_future_match():
            for j, future_frame_matches in enumerate(
                    matches[i + 1 : i + MAX_NUM_INTERPOLATED_FRAMES]):
                for k, future_match in enumerate(future_frame_matches):
                    location2, label2 = future_match
                    if future_match_close(*location2):
                        return j, k, location2, label2

        got = find_future_match()
        if got is None: continue
        j, k, location2, label2 = got
        if label is not None and label2 is None:
            label2 = label
        if label is not None and label2 is not None and label != label2:
            logger.info("Contradicting labels in frames {} and {}".format(i, j))
        if j == 0:
            matches[i + 1][k] = (location2, label2)
        else:
            def interpolate(x): return int(x[0] + (x[1] - x[0]) / (j + 1))
            matches[i + 1].append((
                    tuple(map(interpolate, zip(location, location2))), None))

input_movie.set(cv2.CAP_PROP_POS_FRAMES, 0)

for frame_matches in matches:
    ret, frame1 = input_movie.read()
    if not ret: break
    frame2 = frame1.copy()
    frame_number += 1

    for face_location, label in frame_matches:
        # Draw rectangle for matched face
        top, right, bottom, left = face_location
        cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw rectangle for person's name if recognized
        if label is not None:
             font = cv2.FONT_HERSHEY_DUPLEX
             cv2.putText(frame1, database_persons[label],
                         (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        face = frame2[top:bottom, left:right]
        face = cv2.GaussianBlur(face, (23, 23), 30)
        frame2[top:bottom, left:right] = face

    face_recognized_movie.write(frame1)
    deidentification_movie.write(frame2)

process_duration = datetime.now() - process_start_timepoint
logger.info("Preprocess time: {}".format(preprocess_duration))
logger.info("Process time: {}".format(process_duration))

# All done!
input_movie.release()
cv2.destroyAllWindows()
