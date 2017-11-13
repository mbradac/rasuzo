FROM mislavbradac/rasuzo

COPY rasuzo.py /root/face_recognition/examples/
WORKDIR /root/face_recognition/examples
ENTRYPOINT ["python3", "rasuzo.py", "/video", "/database"]
