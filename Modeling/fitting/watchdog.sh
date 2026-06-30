#!/bin/bash
#SBATCH -p cpu
#SBATCH -t 2-00:00
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -o slurm_out/slurm-watchdog-%j.out
#
# Watches a running PSO array job for a silent stall (a dead worker holding an
# empty tempfile deadlocks the whole array at the per-iteration barrier).
# Alerts via ntfy if no new output appears for STALL_MIN minutes.
#
#   sbatch watchdog.sh <array_job_id> [outdir] [stall_min]

NTFY_TOPIC="pso-cmr-jerryjin-sr0hf95s2v"
JOB_ID="$1"
OUTDIR="${2:-outfiles}"
STALL_MIN="${3:-30}"      # alert if no new output for this many minutes
CHECK_INTERVAL=120        # seconds between checks

notify() {
    curl -s -X POST "https://ntfy.sh/$NTFY_TOPIC" \
        -H "Title: $1" -H "Priority: $2" -d "$3"
}

while true; do
    # Stop watching once no array task is running/pending (notify_done.sh
    # handles the completion/failure summary via the afterany dependency).
    ACTIVE=$(squeue -j "$JOB_ID" -h -t RUNNING,PENDING -o "%i" 2>/dev/null | grep -c .)
    if [ "$ACTIVE" -eq 0 ]; then
        exit 0
    fi

    # Age in seconds of the most recently modified file in OUTDIR.
    NEWEST=$(find "$OUTDIR" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1)
    if [ -n "$NEWEST" ]; then
        NOW=$(date +%s)
        AGE=$(( NOW - ${NEWEST%.*} ))
        if [ "$AGE" -ge $(( STALL_MIN * 60 )) ]; then
            notify "PSO STALLED: job $JOB_ID" "urgent" \
                "No new output in $OUTDIR for $(( AGE / 60 )) min while $ACTIVE task(s) still RUNNING/PENDING. Likely a dead worker holding an empty tempfile. Check: ls -lt $OUTDIR | head"
            # Avoid spamming: back off for one full stall window, then keep
            # watching in case it somehow recovers.
            sleep $(( STALL_MIN * 60 ))
        fi
    fi
    sleep $CHECK_INTERVAL
done
