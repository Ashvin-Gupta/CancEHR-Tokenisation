#!/bin/bash
#$ -cwd                 
#$ -pe smp 4
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
source /data/home/qc25022/CancEHR-Tokenisation/env/bin/activate

# --- Path Definitions ---
CONFIG_FILE="${BASE_DIR}/src/pipelines/config/narrative_config.yaml"
RUN_NAME="narrative_test"

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting narrative pipeline from directory: $(pwd)"
python -m src.narrative.run \
    --config_filepath "${CONFIG_FILE}" 
echo "Narrative pipeline finished."
deactivate
