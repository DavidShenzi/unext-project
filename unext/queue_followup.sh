#!/bin/sh
# The main queue process built its job list at launch, so the stage-2b controls added
# afterwards are invisible to it. Wait for it to exit, then re-run the queue: jobs whose
# model.pth already exists are skipped, so this picks up only what remains.
#
# Process detection uses PowerShell's Get-CimInstance. pgrep does not exist in Git Bash,
# and wmic has been removed from Windows 11 (it returns nothing rather than erroring,
# which would have made this loop exit immediately and silently).
while powershell -NoProfile -Command \
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Select-Object -ExpandProperty CommandLine" \
        2>/dev/null | grep -q "run_queue.py"; do
  sleep 60
done
exec python -u run_queue.py
