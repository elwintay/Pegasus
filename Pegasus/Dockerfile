FROM nvcr.io/nvidia/pytorch:20.03-py3

RUN mkdir -p /Pegasus
WORKDIR /Pegasus

COPY . /Pegasus
WORKDIR /Pegasus

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
