#!/usr/bin/env bash
# Run ONCE after SSH-ing into your AutoDL instance.
# Usage: bash scripts/setup_autodl.sh
set -euo pipefail

echo "=== Setting up AutoDL for YOLO training ==="

# 1. Install ultralytics
pip install ultralytics

# 2. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 3. Extract dataset if needed
if [ ! -d "/root/autodl-tmp/data" ]; then
    ZIP=$(ls /root/autodl-tmp/*.zip 2>/dev/null | head -1)
    if [ -n "$ZIP" ]; then
        echo "Extracting $ZIP ..."
        unzip -o "$ZIP" -d /root/autodl-tmp/
    else
        echo "No zip found in /root/autodl-tmp/. Upload one first."
    fi
fi

# 4. Show dataset summary
if [ -d "/root/autodl-tmp/data" ]; then
    echo ""
    echo "=== Dataset ==="
    for split in train val test; do
        n=$(ls /root/autodl-tmp/data/$split/images/ 2>/dev/null | wc -l)
        echo "  $split: $n images"
    done
    cat /root/autodl-tmp/data/dataset.yaml 2>/dev/null | head -5
fi

echo ""
echo "=== Ready ==="
echo "Run: python /root/autodl-tmp/train.py"
