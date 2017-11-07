# rasuzo

## Steps to run this project on Linux (Ubuntu)

Install docker and git.

Clone this repo: git clone https://github.com/mbradac/rasuzo.git.

Run download_videos.sh to download example videos (from face_recognition repo): ./inputs/download_videos.sh.
Instead of this step you can add your own videos to that folder.

Run run.sh script with root privileges (because of docker): sudo ./run.sh ./inputs/short_hamilton_clip.mp4.
First run of this script will take a long time because it needs to download docker image (1-2GB size).
Every consecutive run will be shorter because docker image will be cached and it won't have to downloaded again.
However, current algorithm is not very fast.
Running face recognition on short_hamilton_clip.mp4 takes 33 seconds on my computer and on hamilton_clip.mp4 almost 5 minutes.
