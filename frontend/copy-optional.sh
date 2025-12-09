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

