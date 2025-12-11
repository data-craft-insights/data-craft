#!/bin/bash
# Cleanup script for Docker containers and images

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Cleaning up Docker resources...${NC}\n"

# Stop and remove test container
if [ "$(docker ps -aq -f name=datacraft-test)" ]; then
    echo -e "Stopping and removing datacraft-test container..."
    docker stop datacraft-test >/dev/null 2>&1 || true
    docker rm datacraft-test >/dev/null 2>&1 || true
    echo -e "${GREEN}✓ Container removed${NC}"
else
    echo -e "No datacraft-test container found"
fi

# Remove image (optional - uncomment if you want to remove the image too)
# if [ "$(docker images -q datacraft-frontend:local)" ]; then
#     echo -e "Removing datacraft-frontend:local image..."
#     docker rmi datacraft-frontend:local
#     echo -e "${GREEN}✓ Image removed${NC}"
# fi

echo -e "\n${GREEN}Cleanup complete!${NC}"

