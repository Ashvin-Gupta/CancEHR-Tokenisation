#!/bin/bash
#$ -cwd                 
#$ -pe smp 1
#$ -l h_rt=1:0:0
#$ -l h_vmem=1G
#$ -o ./logo/
#$ -e ./loge/
#$ -j n

set -e 

module load intel intel-mpi python

source /data/home/qc25022/dask_intel_mpi_venv/bin/activate

# 2. Define file paths
# Use absolute paths to avoid issues
BASE_DIR="/data/home/qc25022/CancEHR-tokeniser"
CONFIG_FILE="${BASE_DIR}/src/pipelines/config/cprd_test.yaml"
RUN_NAME="cprd_test"

# 3. Run the Python script
# We add the --overwrite flag to automatically handle existing directories
echo "Starting tokenization pipeline..."
python -m src.pipelines.run \
    --config_filepath "${CONFIG_FILE}" \
    --run_name "${RUN_NAME}" \
    --overwrite
echo "Pipeline finished."

deactivate