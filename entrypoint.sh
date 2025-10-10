#!/bin/bash

jupyter lab --allow-root --ip=0.0.0.0 --no-browser &
cd tig-benchmarker && python3 slave/main.py