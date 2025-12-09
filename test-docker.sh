#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building Docker Image${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if --no-cache flag should be used
if [ "$1" == "--no-cache" ]; then
    echo -e "${YELLOW}Building without cache...${NC}"
    docker build --no-cache -f frontend/Dockerfile -t datacraft-frontend:local .
else
    docker build -f frontend/Dockerfile -t datacraft-frontend:local .
fi

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Build failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Build successful!${NC}\n"

# Check if container already exists and remove it
if [ "$(docker ps -aq -f name=datacraft-test)" ]; then
    echo -e "${YELLOW}Removing existing container...${NC}"
    docker stop datacraft-test >/dev/null 2>&1 || true
    docker rm datacraft-test >/dev/null 2>&1 || true
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Container${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if gcp directory exists
if [ ! -d "gcp" ]; then
    echo -e "${YELLOW}Warning: gcp/ directory not found. Container will run without mounted credentials.${NC}"
    MOUNT_CMD=""
else
    MOUNT_CMD="-v $(pwd)/gcp:/app/gcp:ro"
fi

docker run -d \
  --name datacraft-test \
  -p 8501:8501 \
  -e GCP_PROJECT_ID=datacraft-data-pipeline \
  -e BQ_DATASET=datacraft_ml \
  -e GCS_BUCKET_NAME=isha-retail-data \
  -e GCP_REGION=us-central1 \
  $MOUNT_CMD \
  datacraft-frontend:local

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Failed to start container!${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Container started successfully!${NC}\n"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Access the app at: http://localhost:8501${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "Useful commands:"
echo -e "  ${YELLOW}View logs:${NC}        docker logs -f datacraft-test"
echo -e "  ${YELLOW}Stop container:${NC}   docker stop datacraft-test"
echo -e "  ${YELLOW}Remove container:${NC} docker stop datacraft-test && docker rm datacraft-test"
echo -e "  ${YELLOW}Shell access:${NC}     docker exec -it datacraft-test /bin/bash"
echo -e "  ${YELLOW}Remove image:${NC}     docker rmi datacraft-frontend:local\n"

