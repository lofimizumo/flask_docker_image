FROM python:3.11-slim-buster

RUN mkdir /notebooks
WORKDIR /notebooks

# Install system dependencies and IPOPT
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    glpk-utils \
    libglpk-dev \
    coinor-libipopt-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /notebooks/
RUN pip install --no-cache-dir -r requirements.txt

# Install Pyomo and its solver interfaces
RUN pip install --no-cache-dir pyomo 

COPY . /notebooks/

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]