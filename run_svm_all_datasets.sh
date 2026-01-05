#!/bin/bash

# Script to run SVM classification for all QSAR datasets
# This script finds all actives/inactives pairs and runs SVM on them

# Don't exit on error - continue with other datasets
set +e

echo "=========================================="
echo "Running SVM for all QSAR datasets"
echo "=========================================="

# Base directories
CONF_DIR="conf/data/qsar"
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-./outputs}"

# Find all actives config files
for actives_config in ${CONF_DIR}/*_actives.yaml; do
    if [ ! -f "$actives_config" ]; then
        continue
    fi
    
    # Extract dataset name (e.g., aid1798 from aid1798_actives.yaml)
    basename_config=$(basename "$actives_config" .yaml)
    dataset_name=$(echo "$basename_config" | sed 's/_actives$//')
    dataset_upper=$(echo "$dataset_name" | tr '[:lower:]' '[:upper:]')
    
    echo ""
    echo "=========================================="
    echo "Processing dataset: $dataset_upper"
    echo "=========================================="
    
    # Try multiple possible paths for prediction files
    # Path 1: Old format in data/qsar/embeddings/
    actives_path1="${DATA_ROOT}/qsar/embeddings/${dataset_upper}/actives/preds_extended.pkl"
    inactives_path1="${DATA_ROOT}/qsar/embeddings/${dataset_upper}/inactives/preds_extended.pkl"
    
    # Path 2: New format in outputs/ (from predict.py) - try mean first (default)
    actives_path2="${OUTPUTS_DIR}/${dataset_name}_actives/predictions_mean.pkl"
    inactives_path2="${OUTPUTS_DIR}/${dataset_name}_inactives/predictions_mean.pkl"
    
    # Path 3: New format in outputs/ (from predict.py) - try median (if aggregation was median)
    actives_path3="${OUTPUTS_DIR}/${dataset_name}_actives/predictions_median.pkl"
    inactives_path3="${OUTPUTS_DIR}/${dataset_name}_inactives/predictions_median.pkl"
    
    # Determine which paths exist
    actives_path=""
    inactives_path=""
    
    if [ -f "$actives_path1" ] && [ -f "$inactives_path1" ]; then
        actives_path="$actives_path1"
        inactives_path="$inactives_path1"
        echo "Using prediction files from: ${DATA_ROOT}/qsar/embeddings/${dataset_upper}/"
    elif [ -f "$actives_path2" ] && [ -f "$inactives_path2" ]; then
        actives_path="$actives_path2"
        inactives_path="$inactives_path2"
        echo "Using prediction files from: ${OUTPUTS_DIR}/ (mean aggregation)"
    elif [ -f "$actives_path3" ] && [ -f "$inactives_path3" ]; then
        actives_path="$actives_path3"
        inactives_path="$inactives_path3"
        echo "Using prediction files from: ${OUTPUTS_DIR}/ (median aggregation)"
    else
        echo "WARNING: Prediction files not found for $dataset_upper"
        echo "  Tried: $actives_path1"
        echo "  Tried: $actives_path2"
        echo "  Tried: $actives_path3"
        echo "  Skipping this dataset..."
        continue
    fi
    
    echo "  Actives: $actives_path"
    echo "  Inactives: $inactives_path"
    
    # Create output directory for this dataset
    svm_output_dir="${OUTPUTS_DIR}/svm_${dataset_name}"
    
    # Run SVM
    echo "Running SVM classification..."
    python -m eddde.svm \
        svm.actives_predictions="${actives_path}" \
        svm.inactives_predictions="${inactives_path}" \
        svm.output_dir="${svm_output_dir}" \
        experiment.name="svm_${dataset_name}"
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Completed SVM for $dataset_upper"
    else
        echo "ERROR: Failed to run SVM for $dataset_upper"
        echo "Continuing with next dataset..."
    fi
done

echo ""
echo "=========================================="
echo "All datasets processed!"
echo "=========================================="
echo "Check the output above for individual dataset results."

