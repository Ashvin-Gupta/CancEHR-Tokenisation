#!/bin/bash
#$ -cwd                 
#$ -pe smp 1
#$ -l h_rt=1:0:0
#$ -l h_vmem=1G
#$ -j n
#$ -o /data/home/qc25022/CancEHR-Tokenisation/HPC_Files/logo/
#$ -e /data/home/qc25022/CancEHR-Tokenisation/HPC_Files/loge/

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/CancEHR-Tokenisation"

# --- Environment Setup ---
module load intel intel-mpi python
source /data/home/qc25022/dask_intel_mpi_venv/bin/activate

# --- Path Definitions ---
CONFIG_FILE="${BASE_DIR}/src/pipelines/config/cprd_test.yaml"
RUN_NAME="cprd_test"

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting tokenization pipeline from directory: $(pwd)"
python -m src.pipelines.run \
    --config_filepath "${CONFIG_FILE}" \
    --run_name "${RUN_NAME}" \
    --overwrite
echo "Pipeline finished."
deactivate