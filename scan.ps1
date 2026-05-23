$pythonExe = "D:\Solo\tradingagents_env\Scripts\python.exe"

if ($args.Count -eq 0) {
    & $pythonExe daily_scan_push.py
} else {
    & $pythonExe daily_scan_push.py @args
}
