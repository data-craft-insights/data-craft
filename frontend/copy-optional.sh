#!/bin/bash
# Don't exit on error - we want to continue even if directories don't exist
set +e

# Copy shared directory if it exists and has content
if [ -d "/tmp/build-context/shared" ] && [ "$(ls -A /tmp/build-context/shared 2>/dev/null)" ]; then
    echo "Copying shared directory..."
    cp -r /tmp/build-context/shared/* /app/shared/ 2>/dev/null || true
else
    echo "shared directory not found or empty, skipping"
fi

# Copy data-pipeline/scripts if it exists and has content
if [ -d "/tmp/build-context/data-pipeline/scripts" ] && [ "$(ls -A /tmp/build-context/data-pipeline/scripts 2>/dev/null)" ]; then
    echo "Copying data-pipeline/scripts..."
    cp -r /tmp/build-context/data-pipeline/scripts/* /app/data-pipeline/scripts/ 2>/dev/null || true
else
    echo "data-pipeline/scripts not found or empty, skipping"
fi

# Copy model_1/v2_vertex if it exists (needed for example_provider)
if [ -d "/tmp/build-context/model_1/v2_vertex" ] && [ "$(ls -A /tmp/build-context/model_1/v2_vertex 2>/dev/null)" ]; then
    echo "Copying model_1/v2_vertex..."
    mkdir -p /app/model_1/v2_vertex
    cp -r /tmp/build-context/model_1/v2_vertex/* /app/model_1/v2_vertex/ 2>/dev/null || true
else
    echo "model_1/v2_vertex not found or empty, skipping"
fi

