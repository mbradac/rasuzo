script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

wget -N https://github.com/ageitgey/face_recognition/raw/master/examples/short_hamilton_clip.mp4
wget -N https://github.com/ageitgey/face_recognition/raw/master/examples/hamilton_clip.mp4
