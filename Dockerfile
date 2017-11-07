FROM mislavbradac/rasuzo

COPY rasuzo.py /root/face_recognition/examples
WORKDIR /root/face_recognition/examples
ENTRYPOINT ["python3", "rasuzo.py", "/inputfile"]
#CMD cd /root/face_recognition/examples && \
#    python3 rasuzo.py
