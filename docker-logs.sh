#!/bin/bash
# View logs from the datacraft-test container

if [ "$(docker ps -aq -f name=datacraft-test)" ]; then
    echo "Following logs for datacraft-test container..."
    echo "Press Ctrl+C to stop"
    echo ""
    docker logs -f datacraft-test
else
    echo "Container 'datacraft-test' not found. Run ./test-docker.sh first."
    exit 1
fi

