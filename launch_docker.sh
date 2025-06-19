#!/bin/sh

type="cpu"
as_root=false

# arguments parser
while [[ $# -gt 0 ]]; do
    case "$1" in
        --as-root)
            as_root=true
            shift
            ;;
        -*)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "$type_set" ]]; then
                type="$1"
                type_set=true
                shift
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

OVERRIDE=""
if [ "$as_root" = true ]; then
	echo "Running as root"
	export DOCKER_UID=root
	export DOCKER_GID=root
else
	echo "Running as non-root"
	export DOCKER_UID=$(id -u)
	export DOCKER_GID=$(id -g)
	OVERRIDE="-f docker/docker-compose.override.map-host-account.yml"
fi

if [ "$type" = "cpu" ]; then
	echo "Running CPU version"
elif [ "$type" = "nvidia" ]; then
	echo "Running NVIDIA version"
	OVERRIDE="$OVERRIDE -f docker/docker-compose.override.nvidia.yml"
elif [ "$type" = "amdgpu" ]; then
	echo "Running AMD GPU version"
	OVERRIDE="$OVERRIDE -f docker/docker-compose.override.amdgpu.yml"
elif [ "$type" = "intel-gpu" ]; then
	echo "Running Intel GPU version"
	OVERRIDE="$OVERRIDE -f docker/docker-compose.override.intel-gpu.yml"
else
	echo "Usage: $0 [cpu|nvidia|amdgpu|intel-gpu]"
	exit 1
fi

# launch docker in detached mode
docker compose -f docker-compose.yml $OVERRIDE up -d --no-build
