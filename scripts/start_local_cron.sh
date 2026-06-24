#!/bin/sh
set -eu

mkdir -p /app/logs /app/results

cat >/etc/cron.d/tradingagents <<'EOF'
SHELL=/bin/sh
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

30 13 * * 1-5 root cd /app && python -u scripts/run_scan_pipeline.py --mode intraday --retries 3 --base-sleep 20 >> /app/logs/intraday.log 2>&1
30 17 * * 1-5 root cd /app && python -u scripts/run_scan_pipeline.py --mode after_hours --retries 3 --base-sleep 20 >> /app/logs/after_hours.log 2>&1
EOF

chmod 0644 /etc/cron.d/tradingagents
crontab /etc/cron.d/tradingagents

echo "Local scanner cron installed:"
cat /etc/cron.d/tradingagents

exec cron -f
