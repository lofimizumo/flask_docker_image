FROM paperspace/gradient-base:pt112-tf29-jax0317-py39-20230125

RUN mkdir /notebooks
WORKDIR /notebooks
COPY . /notebooks/

# ENV CONDA_ALWAYS_YES=true

# RUN mamba install -c conda-forge -c pytorch u8darts-all
RUN pip install darts
RUN pip install wandb

# The code to run when container is started
COPY . .
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]