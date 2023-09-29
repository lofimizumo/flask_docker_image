FROM --platform=$BUILDPLATFORM python:3.10-alpine AS builder

RUN mkdir /notebooks
WORKDIR /notebooks
COPY . /notebooks/

# ENV CONDA_ALWAYS_YES=true

RUN pip install flask
RUN pip install pandas
RUN pip install requests
RUN pip install numpy

# The code to run when container is started
COPY . .
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]