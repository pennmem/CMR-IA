#!/bin/bash
#
# Single-shot stall check for a running PSO array job (a dead worker holding
# an empty tempfile deadlocks the whole array at the per-iteration barrier).
# Alerts via ntfy if no new output has appeared for STALL_MIN minutes.
#
# Meant to be run periodically via `scrontab`, not submitted directly with
# sbatch -- Slurm rejects long-lived jobs that just sleep/poll in a loop.
# `pgo` registers this via `scrontab` automatically after launching an array
# job, and `notify_done.sh` removes the entry once that job finishes (via
# its afterany dependency), so no manual scrontab bookkeeping is needed for
# jobs started through `pgo`.
#
#   ./watchdog.sh <array_job_id> [outdir] [stall_min]
#
# To wire it up manually for a job not started via `pgo` (scrontab -e),
# checking every 2 minutes:
#
#   #SCRON -p service
#   #SCRON -t 00:02:00
#   #SCRON -D /absolute/path/to/job/dir
#   #SCRON -o slurm_out/slurm-watchdog-%j.out
#   #SCRON --open-mode=append
#   #SCRON --dependency=singleton
#   */2 * * * * /path/to/watchdog.sh <array_job_id> /absolute/path/to/job/dir/outfiles 30

NTFY_TOPIC="pso-cmr-jerryjin-sr0hf95s2v"
JOB_ID="$1"
OUTDIR="${2:-outfiles}"
STALL_MIN="${3:-30}"      # alert if no new output for this many minutes
STATE_FILE="$OUTDIR/.watchdog_last_alert"

notify() {
    curl -s -X POST "https://ntfy.sh/$NTFY_TOPIC" \
        -H "Title: $1" -H "Priority: $2" -d "$3"
}

# Nothing left to watch; notify_done.sh already covers completion/failure.
ACTIVE=$(squeue -j "$JOB_ID" -h -t RUNNING,PENDING -o "%i" 2>/dev/null | grep -c .)
if [ "$ACTIVE" -eq 0 ]; then
    exit 0
fi

# Age in seconds of the most recently modified file in OUTDIR.
NEWEST=$(find "$OUTDIR" -maxdepth 1 -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1)
if [ -z "$NEWEST" ]; then
    exit 0
fi

NOW=$(date +%s)
AGE=$(( NOW - ${NEWEST%.*} ))

if [ "$AGE" -lt $(( STALL_MIN * 60 )) ]; then
    # Healthy again -- clear any prior alert cooldown so a future stall
    # notifies promptly instead of waiting out the old backoff window.
    rm -f "$STATE_FILE"
    exit 0
fi

# Stalled. Avoid spamming: only alert once per STALL_MIN window.
LAST_ALERT=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
if [ $(( NOW - LAST_ALERT )) -ge $(( STALL_MIN * 60 )) ]; then
    notify "PSO STALLED: job $JOB_ID" "urgent" \
        "No new output in $OUTDIR for $(( AGE / 60 )) min while $ACTIVE task(s) still RUNNING/PENDING. Likely a dead worker holding an empty tempfile. Check: ls -lt $OUTDIR | head"
    echo "$NOW" > "$STATE_FILE"
fi
