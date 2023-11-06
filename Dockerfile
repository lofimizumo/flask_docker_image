FROM --platform=$BUILDPLATFORM python:3.11-alpine AS builder

RUN mkdir /notebooks
WORKDIR /notebooks
COPY . /notebooks/

# ENV CONDA_ALWAYS_YES=true

RUN pip install numpy==1.25.1
RUN pip install flask
RUN pip install requests
RUN pip install pytz

# The code to run when container is started
COPY . .
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]