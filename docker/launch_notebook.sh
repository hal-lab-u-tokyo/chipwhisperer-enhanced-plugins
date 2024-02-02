#!/bin/sh

jupyter notebook --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser --allow-root --NotebookApp.token=${JUPYTER_TOKEN}