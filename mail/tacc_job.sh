#!/bin/bash
#SBATCH -p skx-dev
#SBATCH -J citylearn
#SBATCH -N 4
#SBATCH --tasks-per-node 1
#SBATCH -t 02:00:00
#SBATCH --mail-user=nweye@utexas.edu
#SBATCH --mail-type=all
#SBATCH -o /work2/07083/ken658/projects/citylearn/buildsys_2022/log/tacc_job.out
#SBATCH -A DemandAnalysis

# load modules
module load launcher

# activate virtual environment
source /work2/07083/ken658/projects/citylearn/buildsys_2022/env/bin/activate

# set launcher environment variables
export LAUNCHER_WORKDIR="/work2/07083/ken658/projects/citylearn/buildsys_2022"
export LAUNCHER_JOB_FILE="/work2/07083/ken658/projects/citylearn/buildsys_2022/job/work_order/work_order.sh"

${LAUNCHER_DIR}/paramrun
