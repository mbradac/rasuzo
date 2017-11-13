# rasuzo

## Steps to run this project on Linux (Ubuntu)

Install docker and git.

Clone this repo: git clone https://github.com/mbradac/rasuzo.git.

Run run.sh script with root privileges (because of docker): sudo ./run.sh ./inputs/will_ferrell_chad_smith/medium_15s.webm ./inputs/will_ferrell_chad_smith/database/.
First parameter is path to the video and second to the database of faces. Take a look at ./inputs/will_ferell_chad_smith/database/ folder to see what should be the hierarchy of database folder.
Output of the script will be in subfolder of outputs folder.
First run of this script will take a long time because it needs to download docker image (~2GB size).
Every consecutive run will be shorter because docker image will be cached and it won't have to downloaded again.
However, current algorithm is not very fast at all.
