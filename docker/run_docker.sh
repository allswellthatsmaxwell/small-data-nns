#!/bin/bash
docker pull gaarv/jupyter-keras:latest
docker run -it --rm \
       -p 8888:8888 \
       -v /home/maxwell/projects:/home/jovyan/projects \
        --user 1001 --group-add users \
       --name $USER \
       -e GRANT_SUDO=yes \
       -e JUPYTER_ENABLE_LAB=yes \
       gaarv/jupyter-keras:latest jupyter lab \
       --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
