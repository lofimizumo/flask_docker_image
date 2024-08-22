FROM python:3.11-slim-buster

RUN mkdir /notebooks
WORKDIR /notebooks

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    git \
    pkg-config \
    wget \
    make \
    liblapack-dev \
    libblas-dev \
    libmetis-dev \
    glpk-utils \
    libglpk-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and run CoinBrew to install IPOPT
RUN wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
RUN chmod u+x coinbrew
RUN ./coinbrew fetch Ipopt --no-prompt
RUN ./coinbrew build Ipopt --prefix=/usr/local --test --no-prompt --verbosity=3
RUN rm -rf build

# Set environment variables
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV PATH="/usr/local/bin:${PATH}"

# Copy requirements and install Python packages
COPY requirements.txt /notebooks/
RUN pip install --no-cache-dir -r requirements.txt

# Install Pyomo and solvers
RUN pip install --no-cache-dir pyomo
RUN pip install --no-cache-dir glpk
RUN pip install --no-cache-dir cyipopt

COPY . /notebooks/

# Add diagnostic commands
RUN echo "Checking IPOPT installation:" && \
    which ipopt && \
    ldd $(which ipopt) && \
    echo "Checking GLPK installation:" && \
    which glpsol && \
    ldd $(which glpsol) && \
    echo "Checking cyipopt installation:" && \
    python -c "import cyipopt; print(cyipopt.__file__)" && \
    echo "Checking Pyomo solvers:" && \
    python -c "from pyomo.environ import *; print('IPOPT available:', SolverFactory('ipopt').available()); print('GLPK available:', SolverFactory('glpk').available())"

EXPOSE 5000

ENTRYPOINT ["python", "-u", "app.py"]