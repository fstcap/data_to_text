FROM tensorflow/tensorflow:2.10.1-gpu

WORKDIR /app

COPY requirements-linux.txt /app/requirements-linux.txt

RUN pip install -U pip && pip install -r /app/requirements-linux.txt

# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

# ENTRYPOINT ["python", "/app/main.py"]
