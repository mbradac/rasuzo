import face_recognition
import cv2
import sys
import os
import logging
from datetime import datetime
from argparse import ArgumentParser
from collections import Counter
from sklearn.linear_model import LogisticRegression

MAX_NUM_INTERPOLATED_FRAMES = 200
MATCH_CONFIDENCE_THRESHOLD = 0.0
INTERPOLATION_MATCH_THRESHOLD_RATIO = 0.5
NORM_RATIO_THRESHOLD = 9.0
NORM_THRESHOLD = 10000
CHAIN_LENGTH_THRESHOLD = 10
INTERPOLATE_TO_BOUNDARY_THRESHOLD = 100

logging.basicConfig(level="INFO")
logger = logging.getLogger("lab2")


# Finds frames in which scene changes. Returns [int] each element being frame
# number (zero index) on which change happens.
def find_scene_changes(video):
    start_time = datetime.now()
    prev_frame = None
    prev_norm = None
    frame_num = 0
    scene_changes = []

    while True:
        ret, frame = video.read()
        frame_num += 1
        if not ret: break
        frame = cv2.GaussianBlur(frame, (23, 23), 30)
        if prev_frame is not None:
            norm = cv2.norm(prev_frame, frame)
            if prev_norm is not None:
                if abs(prev_norm - norm) / prev_norm > NORM_RATIO_THRESHOLD \
                        and norm > NORM_THRESHOLD:
                    logger.info("Found new scene at frame {}".format(frame_num))
                    scene_changes.append(frame_num - 1)
            prev_norm = norm
        prev_frame = frame
        font = cv2.FONT_HERSHEY_DUPLEX

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    duration = datetime.now() - start_time
    logger.info("Find scene changes time {}".format(duration))
    return scene_changes


# Finds matches across multiple frames that potentially belong to the same
# continual match. Returns [[(frame_number, location, label)]] - list of chains
# each chain being a list of matches.
def groups_matches_to_chains(matches):
    match_chains = []
    match_in_chain = {}

    for i, frame_matches in enumerate(matches):
        for in_frame_id, match in enumerate(frame_matches):
            location, label = match
            top, right, bottom, left = location
            width, height = abs(right - left), abs(top - bottom)

            chain_id = match_in_chain.get((i, in_frame_id), None)
            if chain_id is None:
                chain_id = len(match_chains)
                match_in_chain[(i, in_frame_id)] = chain_id
                match_chains.append([(i, location, label)])

            def future_match_close(top2, right2, bottom2, left2):
                threshold = INTERPOLATION_MATCH_THRESHOLD_RATIO
                if abs(top2 - top) / height > threshold:
                    return False
                if abs(bottom2 - bottom) / height > threshold:
                    return False
                if abs(left2 - left) / width > threshold:
                    return False
                if abs(right2 - right) / width > threshold:
                    return False
                return True

            def find_future_match():
                for j, future_frame_matches in enumerate(
                        matches[i + 1 : i + MAX_NUM_INTERPOLATED_FRAMES]):
                    for k, future_match in enumerate(future_frame_matches):
                        location2, label2 = future_match
                        if future_match_close(*location2):
                            return i + 1 + j, k, location2, label2

            got = find_future_match()
            if got is None: continue
            j, k, location2, label2 = got
            match_in_chain[(j, k)] = chain_id
            match_chains[chain_id].append((j, location2, label2))

    return match_chains


def interpolate_faces(matches):
    match_chains = groups_matches_to_chains(matches)
    interpolated_matches = [[] for _ in range(len(matches))]

    for chain in match_chains:
        frame_number0, location0, label0 = chain[0]
        frame_numbern, locationn, labeln = chain[-1]
        if frame_numbern - frame_number0 < CHAIN_LENGTH_THRESHOLD: continue
        if frame_number0 < INTERPOLATE_TO_BOUNDARY_THRESHOLD:
            frame_number0, label0 = 0, None
            chain.insert(0, (frame_number0, location0, label0))
        if len(matches) - frame_numbern < INTERPOLATE_TO_BOUNDARY_THRESHOLD:
            frame_numbern, labeln = len(matches) - 1, None
            chain.append((frame_numbern, locationn, labeln))

        labels = list(filter(None, map(lambda x: x[2], chain)))
        if labels != []:
            counter = Counter(labels)
            label, count = counter.most_common(1)[0]
            if count != len(labels):
                logger.warning(
                        "Contradicting labels {} in chain in frames {}-{}"
                        .format(counter.items(), frame_number0, frame_numbern))
        else:
            label = None

        interpolated_matches[frame_number0].append((location0, label))
        for prev_match, match in zip(chain[:-1], chain[1:]):
            frame_number1, location1, label1 = prev_match
            frame_number2, location2, label2 = match
            for i in range(frame_number1 + 1, frame_number2):
                interpolate = lambda x: int(x[0] + (x[1] - x[0]) /
                        (frame_number2 - frame_number1) * (i - frame_number1))
                interpolated_matches[i].append((tuple(map(interpolate,
                        zip(location1, location2))), label))
            interpolated_matches[frame_number2].append((location2, label))

            #match_chains[chain_id].append((j, k))
            #    if (i + MAX_NUM_INTERPOLATED_FRAMES <= len(matches) or
            #            i + 1 == len(matches)) : continue
            #    got = i + 1, None, location
            #j, k, location2 = got
#            if label is not None and label2 is None:
#                label2 = label
#            if label is not None and label2 is not None and label != label2:
#                logger.info("Contradicting labels in frames {} and {}".format(i, j))
#            if j == 0:
#                matches[i + 1][k] = (location2, label2)
#            else:
    return interpolated_matches


parser = ArgumentParser()
parser.add_argument('video_path')
parser.add_argument('database_path')
parser.add_argument('--no-interpolate', action='store_true')
args = parser.parse_args()

# Open the input movie file
input_movie = cv2.VideoCapture(args.video_path)
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
for label, person_name in enumerate(os.listdir(args.database_path)):
    person_folder_path = os.path.join(args.database_path, person_name)
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
scene_changes = find_scene_changes(input_movie)

# Find faces in each frame of video
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    # Quit when the input video file ends
    if not ret: break
    frame_number += 1
    logger.info("Finding matches in frame {} / {}".format(frame_number, length))

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

input_movie.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Interpolate face matches
if not args.no_interpolate:
    scene_start = 0
    for i, scene_end in enumerate(scene_changes):
        logger.info("Interpolating scene {}".format(i))
        matches[scene_start:scene_end] = interpolate_faces(
                matches[scene_start:scene_end])
        scene_start = scene_end

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
