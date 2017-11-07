#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run.sh input_file_path"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t video_face_recognition ${script_dir}

input_path=$(readlink -f "$1")
docker run -v "$input_path:/inputfile" video_face_recognition

input_path_basename="$(echo ${1##*/})"
output_filename="$(date +%Y-%m-%d-%H-%M-%S)_$(echo ${input_path_basename%.*}).avi"
docker cp "$(docker ps -lq):/root/face_recognition/examples/output.avi" "$output_filename"
