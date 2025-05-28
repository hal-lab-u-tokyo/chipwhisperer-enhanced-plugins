#!/bin/sh

# get the UID and GID of the current user
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# check if root
NOTEBOOK_ARGS="--ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser --NotebookApp.token=${JUPYTER_TOKEN}"

if [ "$USER_ID" -eq 0 ]; then
	NOTEBOOK_ARGS="${NOTEBOOK_ARGS} --allow-root"
else
	# create temporary home directory
	export HOME=/var/tmp/home_${USER_ID}
	mkdir -p "$HOME"
	export XDG_RUNTIME_DIR="$HOME/.xdg"
	mkdir -p "$XDG_RUNTIME_DIR"
	echo "export HOME=${HOME}" >> /var/tmp/init/${USER_ID}_bashrc

	# prepare directories for jupyter
	mkdir -p "$HOME/.local"
	export PATH="$HOME/.local/bin:$PATH"
fi

# no commad passed
if [ $# -eq 0 ]; then
	echo "Starting Jupyter Notebook..."
	jupyter notebook $NOTEBOOK_ARGS
else
	exec "$@"
fi
