FROM --platform=$BUILDPLATFORM python:3.11-alpine AS builder

RUN mkdir /notebooks
WORKDIR /notebooks

# Install system dependencies
RUN apk update && \
    apk --no-cache add \
    linux-headers \
    gcc \
    g++ \
    glpk \
    glpk-dev

COPY requirements.txt /notebooks/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /notebooks/

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]