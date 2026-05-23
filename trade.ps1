$env:PATH = "D:\Solo\tradingagents_env\Scripts;" + $env:PATH
$pythonExe = "D:\Solo\tradingagents_env\Scripts\python.exe"

if ($args.Count -eq 0) {
    & $pythonExe live_trader.py
} else {
    & $pythonExe live_trader.py @args
}
