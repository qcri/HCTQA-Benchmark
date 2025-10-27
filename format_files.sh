#!/bin/bash

set -e  # stop if any command fails

echo "===== INFLATING REAL-WORLD TABLES ====="

DATASET_DIR="./datasets/realWorld_datasets/tables"

# destination folders
IMAGES_DEST="$DATASET_DIR/images"
CSVS_DEST="$DATASET_DIR/csvs"
HTMLS_DEST="$DATASET_DIR/htmls"
MDS_DEST="$DATASET_DIR/mds"

# create folders
mkdir -p "$IMAGES_DEST" "$CSVS_DEST" "$HTMLS_DEST" "$MDS_DEST"

# --- 1. Extract all image parts and move into one folder ---
for file in "$DATASET_DIR"/images_pt*.tar.gz; do
    echo "Extracting $file ..."
    tar -xzf "$file" -C "$DATASET_DIR"
done

# move all image files to one folder
find "$DATASET_DIR" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) -exec mv {} "$IMAGES_DEST" \;
# remove empty extracted dirs
find "$DATASET_DIR" -mindepth 1 -type d -name "images_pt*" -exec rm -r {} + || true

# --- 2. Extract csvs, htmls, and mds archives ---
for type in csvs htmls mds; do
    FILE="$DATASET_DIR/${type}.tar.gz"
    DEST_VAR="${type^^}_DEST" # uppercase variable name, e.g. CSVS_DEST
    DEST=${!DEST_VAR}
    if [ -f "$FILE" ]; then
        echo "Extracting $FILE ..."
        tar -xzf "$FILE" -C "$DEST"
        # handle nested folder issue like csvs/csvs
        if [ -d "$DEST/$type" ]; then
            mv "$DEST/$type"/* "$DEST" && rm -r "$DEST/$type"
        fi
    fi
done

echo "✅ Real-world tables inflated."

# --- 3. Inflate real-world QAPs ---
QAPS_DIR="./datasets/realWorld_datasets/qaps"
QAPS_FILE="$QAPS_DIR/realWorld_HCT_qaps.json.gz"
if [ -f "$QAPS_FILE" ]; then
    echo "Inflating $QAPS_FILE ..."
    gunzip -kf "$QAPS_FILE"
fi

echo "✅ Real-world QAPs inflated."

# --- 4 & 5. Inflate synthetic datasets ---
for SYN_TYPE in original text_obfuscated; do
    SYN_PATH="./datasets/synthetic_datasets/$SYN_TYPE"
    echo "===== INFLATING SYNTHETIC DATASET: $SYN_TYPE ====="
    for type in csvs htmls mds; do
        FILE="$SYN_PATH/${type}.tar.gz"
        DEST="$SYN_PATH/$type"
        mkdir -p "$DEST"
        if [ -f "$FILE" ]; then
            echo "Extracting $FILE ..."
            tar -xzf "$FILE" -C "$DEST"
            if [ -d "$DEST/$type" ]; then
                mv "$DEST/$type"/* "$DEST" && rm -r "$DEST/$type"
            fi
        fi
    done
done

echo "✅ Synthetic datasets inflated."

# --- 6. Inflate model responses ---
RESP_DIR="./results/model_responses_for_experiments_in_paper"
echo "===== INFLATING MODEL RESPONSE FILES ====="
for file in "$RESP_DIR"/*.tar.gz; do
    [ -e "$file" ] || continue
    BASENAME=$(basename "$file" .tar.gz)
    DEST="$RESP_DIR/$BASENAME"
    mkdir -p "$DEST"
    echo "Extracting $file into $DEST ..."
    tar -xzf "$file" -C "$DEST"
done

echo "✅ Model responses inflated."
echo "===== ALL DONE ====="
