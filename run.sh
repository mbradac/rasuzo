#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh video_file_path database_folder_path"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t video_face_recognition ${script_dir}

video=$(readlink -f "$1")
database=$(readlink -f "$2")
docker run -v "$video:/video" -v "$database:/database" video_face_recognition

input_path_basename="$(echo ${1##*/})"
output_folder="$script_dir/outputs/$(date +%Y-%m-%d-%H-%M-%S)_${input_path_basename%.*}"
docker cp "$(docker ps -lq):/root/face_recognition/examples/face_recognized.avi" "$output_folder"
docker cp "$(docker ps -lq):/root/face_recognition/examples/deidentification.avi" "$output_folder"
