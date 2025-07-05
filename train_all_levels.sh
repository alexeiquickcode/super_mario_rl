#!/bin/bash

set -e
echo "Training all levels..."
. .venv/bin/activate

for world in {1..8}; do
    for stage in {1..4}; do
        echo "Training World $world - Stage $stage..."
        python train.py --world $world --stage $stage
        echo "Completed World $world - Stage $stage"
        echo ""
    done
done

echo "Finished training all levels" 