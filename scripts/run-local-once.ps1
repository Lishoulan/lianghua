$ErrorActionPreference = "Stop"

$mode = if ($args.Count -gt 0) { $args[0] } else { "intraday" }
if ($mode -notin @("intraday", "after_hours")) {
    throw "Mode must be intraday or after_hours."
}

docker exec tradingagents-local-scanner python -u scripts/run_scan_pipeline.py --mode $mode --retries 3 --base-sleep 20
