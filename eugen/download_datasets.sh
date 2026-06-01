#!/bin/bash
#
# download_datasets.sh
# Run this on the DelftBlue login node (with internet access).
# Downloads validation COCO dataset, weights, and sets up Gen1/N-Cars directories.
#
# Usage:
#   bash eugen/download_datasets.sh [--gen1-url URL] [--ncars-url URL]

set -euo pipefail

GEN1_URL="https://1-34-v3-10.download.kdrive.infomaniakusercontent.com/2/download/019e7450-3337-790e-8b37-04fcb1dd6ddd/sharing_archive"
NCARS_URL="https://kdrive.infomaniak.com/app/share/975517/eb418265-0d5a-43a7-b87e-b3d785f17292/files/148/download"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gen1-url)
            GEN1_URL="$2"
            shift 2
            ;;
        --ncars-url)
            NCARS_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

DATASET_DIR="/scratch/$USER/datasets"
echo "Initializing dataset directory in scratch: $DATASET_DIR"
mkdir -p "$DATASET_DIR"

# Ensure symlink from project space to scratch exists
PROJECT_DIR="$HOME/EMS-YOLO"
if [ -d "$PROJECT_DIR" ]; then
    echo "Creating symlink for datasets..."
    ln -sfn "$DATASET_DIR" "$PROJECT_DIR/datasets"
else
    echo "Warning: Project directory $PROJECT_DIR not found. Skipping symlink creation."
fi

# 1. COCO 2017 Validation Set
if [ ! -d "$DATASET_DIR/coco" ] || [ -z "$(ls -A "$DATASET_DIR/coco")" ]; then
    echo "=== Downloading COCO 2017 Validation Set ==="
    mkdir -p "$DATASET_DIR/coco"
    cd "$DATASET_DIR/coco"
    wget -c http://images.cocodataset.org/zips/val2017.zip
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    
    echo "Extracting COCO val2017 and annotations..."
    unzip -q val2017.zip
    unzip -q annotations_trainval2017.zip
    rm -f val2017.zip annotations_trainval2017.zip
else
    echo "=== COCO 2017 Validation Set already exists. Skipping. ==="
fi

# 2. Pretrained COCO weights (best.pt and last.pt from authors)
WEIGHTS_DIR="$PROJECT_DIR/runs/train/exp/weights"
if [ ! -f "$WEIGHTS_DIR/best.pt" ] || [ ! -f "$WEIGHTS_DIR/last.pt" ]; then
    echo "=== Downloading COCO Pretrained Weights ==="
    mkdir -p "$WEIGHTS_DIR"
    cd "$WEIGHTS_DIR"
    python3 -c "
import requests, re, os
def dl(fid, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000000:
        print(f'{path} already exists. Skipping.')
        return
    s = requests.Session()
    r = s.get('https://drive.google.com/uc?export=download&id=' + fid)
    if 'Virus scan' in r.text or 'can\'t scan' in r.text:
        inputs = re.findall(r'<input type=\"hidden\" name=\"([^\"]+)\" value=\"([^\"]+)\"', r.text)
        params = {n: v for n, v in inputs}
        action_match = re.search(r'action=\"([^\"]+)\"', r.text)
        action_url = action_match.group(1) if action_match else 'https://drive.usercontent.google.com/download'
        r = s.get(action_url, params=params, stream=True)
    else:
        r = s.get('https://drive.google.com/uc?export=download&id=' + fid, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(32768):
            if chunk: f.write(chunk)
    print(f'Downloaded {path}, size: {os.path.getsize(path)} bytes')
dl('1AWIVMcn9-VzuQG84y3aQEvHcfi5zVxz0', 'best.pt')
dl('1VVXQFOJ2Cgv5tlcOCNhai11M3_RViqpb', 'last.pt')
"
else
    echo "=== COCO Pretrained Weights already exist. Skipping. ==="
fi

# 3. Gen1 Pre-computed dataset from kDrive
if [ ! -d "$DATASET_DIR/gen1_processed" ] || [ -z "$(ls -A "$DATASET_DIR/gen1_processed")" ]; then
    if [ -z "$GEN1_URL" ]; then
        echo ""
        echo "=== Gen1 Pre-computed Dataset Setup ==="
        echo "Please provide a valid direct download URL."
        echo "Default: https://1-34-v3-10.download.kdrive.infomaniakusercontent.com/2/download/019e7450-3337-790e-8b37-04fcb1dd6ddd/sharing_archive"
        echo -n "Paste the Gen1 direct download link (or press enter to use default): "
        read -r INPUT_URL || true
        GEN1_URL="${INPUT_URL:-$GEN1_URL}"
    fi
    
    if [ -n "$GEN1_URL" ]; then
        echo "Downloading Gen1 pre-computed dataset..."
        cd "$DATASET_DIR"
        wget -c -O gen1_precomputed.zip "$GEN1_URL"
        echo "Extracting Gen1 pre-computed dataset..."
        unzip -q gen1_precomputed.zip
        rm -f gen1_precomputed.zip
        
        # Robust check to ensure the folder is named gen1_processed
        if [ ! -d "gen1_processed" ]; then
            if [ -d "train" ] && [ -d "val" ]; then
                echo "Moving train/val/test folders into gen1_processed..."
                mkdir -p gen1_processed
                mv train val test gen1_processed/ || true
            else
                for d in */; do
                    d_clean="${d%/}"
                    if [ "$d_clean" != "ncars_raw" ] && [ "$d_clean" != "coco" ] && [ "$d_clean" != "gen1_processed" ]; then
                        echo "Renaming extracted Gen1 folder $d_clean to gen1_processed..."
                        mv "$d_clean" "gen1_processed"
                        break
                    fi
                done
            fi
        fi
    fi
else
    echo "=== Gen1 Pre-processed Dataset already exists. Skipping. ==="
fi

# 4. N-Cars dataset from kDrive
if [ ! -d "$DATASET_DIR/ncars_raw" ] || [ -z "$(ls -A "$DATASET_DIR/ncars_raw")" ]; then
    if [ -z "$NCARS_URL" ]; then
        echo ""
        echo "=== N-Cars Dataset Setup ==="
        echo "Please provide a valid direct download URL."
        echo "Default: https://kdrive.infomaniak.com/app/share/975517/eb418265-0d5a-43a7-b87e-b3d785f17292/files/148/download"
        echo -n "Paste the N-Cars direct download link (or press enter to use default): "
        read -r INPUT_URL || true
        NCARS_URL="${INPUT_URL:-$NCARS_URL}"
    fi
    
    if [ -n "$NCARS_URL" ]; then
        echo "Downloading N-Cars dataset..."
        cd "$DATASET_DIR"
        wget -c -O Prophesee_Dataset_n_cars.7z "$NCARS_URL"
        echo "Extracting N-Cars dataset..."
        mkdir -p ncars_raw
        
        # Try extracting with py7zr first, fallback to standard 7z/7za if available
        if command -v 7z &> /dev/null; then
            7z x -y Prophesee_Dataset_n_cars.7z -oncars_raw > /dev/null
        elif command -v 7za &> /dev/null; then
            7za x -y Prophesee_Dataset_n_cars.7z -oncars_raw > /dev/null
        else
            echo "Using py7zr via uv to extract .7z archive..."
            uv run --with py7zr python3 -c "import py7zr; py7zr.SevenZipFile('Prophesee_Dataset_n_cars.7z', mode='r').extractall(path='ncars_raw')"
        fi
        
        rm -f Prophesee_Dataset_n_cars.7z
        
        # If it extracted into a subdirectory, move it to the root of ncars_raw
        if [ -d "ncars_raw/Prophesee_Dataset_n_cars" ]; then
            echo "Moving extracted N-Cars files to ncars_raw root..."
            mv ncars_raw/Prophesee_Dataset_n_cars/* ncars_raw/ || true
            rmdir ncars_raw/Prophesee_Dataset_n_cars || true
        fi
    fi
else
    echo "=== N-Cars Raw Dataset already exists. Skipping. ==="
fi


echo "=== Dataset setup complete ==="
