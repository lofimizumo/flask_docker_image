FROM --platform=$BUILDPLATFORM python:3.11-alpine AS builder

RUN mkdir /notebooks
WORKDIR /notebooks

COPY requirements.txt /notebooks/

RUN apk update && \
    apk --no-cache add linux-headers gcc g++ && \
    pip install --no-cache-dir -r requirements.txt

COPY . /notebooks/

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]