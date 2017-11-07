import face_recognition
import cv2
import sys

assert len(sys.argv) == 2, "Usage: python3 rasuzo.py input_file_path"

# Open the input movie file
input_movie = cv2.VideoCapture(sys.argv[1])
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_movie.get(cv2.CAP_PROP_FPS)
print("Input video width = {}, height = {}, fps = {}, number of frames = {}"
        .format(width, height, fps, length))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

face_locations = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)

    # Label the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
