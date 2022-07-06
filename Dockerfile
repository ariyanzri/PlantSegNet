FROM jupyter/tensorflow-notebook:2021-09-07

COPY ["data", "models", "notebooks", "train_and_inference"] /work/repos/SorghumPartNet 

COPY "dataset" "/space/ariyanzarei/sorghum_segmentation/dataset/2022-03-10/"
COPY "models" "/space/ariyanzarei/sorghum_segmentation/models"


USER root


RUN cd /work/repos/SorghumPartNet && \
    pip install -r  \
      zipcodes \
      pymongo \
      dask-mongo \
      pytz \
      tqdm

ENV PYTHONPATH=.:/home/jovyan/working

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID