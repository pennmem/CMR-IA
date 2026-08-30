#!/bin/bash
#SBATCH -p normal
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

# Drop this job's watchdog scrontab entry (and its #SCRON header block)
# now that the array is done.
EXISTING=$(scrontab -l 2>/dev/null)
if [ $? -ne 0 ]; then EXISTING=""; fi
printf '%s\n' "$EXISTING" | awk -v marker="watchdog.sh $JOB_ID " '
    /^#SCRON/ { buf[n++]=$0; next }
    {
        if (index($0, marker) > 0) { n=0; next }
        for (i=0;i<n;i++) print buf[i]
        n=0
        print $0
    }
    END { for (i=0;i<n;i++) print buf[i] }
' | scrontab -
