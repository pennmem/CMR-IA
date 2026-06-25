#!/bin/bash
#SBATCH -p cpu
#SBATCH -t 0-00:05
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -o slurm_out/slurm-notify-%j.out

NTFY_TOPIC="pso-cmr-jerryjin-sr0hf95s2v"
JOB_ID="$1"

ENTRIES=$(sacct -j "$JOB_ID" --format=JobID,State --noheader --parsable2 \
    | grep -E "^[0-9]+(_[0-9]+)?\|" | grep -v "\.")

TOTAL=$(echo "$ENTRIES" | grep -c .)
COMPLETED=$(echo "$ENTRIES" | grep -c "|COMPLETED")
FAILED=$((TOTAL - COMPLETED))

if [ "$FAILED" -eq 0 ]; then
    curl -s -X POST "https://ntfy.sh/$NTFY_TOPIC" \
        -H "Title: Job $JOB_ID done" \
        -H "Priority: default" \
        -d "All $TOTAL tasks completed successfully"
else
    curl -s -X POST "https://ntfy.sh/$NTFY_TOPIC" \
        -H "Title: Job $JOB_ID done" \
        -H "Priority: high" \
        -d "$FAILED/$TOTAL tasks FAILED"
fi
