#!/bin/bash
sleep 10
# Run the Python script
while true; do
    python3 hm3d-online/goat-nav.py
    if [ $? -eq 0 ]; then
        break
    fi
done        