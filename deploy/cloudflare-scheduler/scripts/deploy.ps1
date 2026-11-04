<#
    远程部署 Cloudflare Worker（lianghua 定时触发器）

    用法（在 deploy/cloudflare-scheduler 目录下）：
        pwsh -File scripts/deploy.ps1

    做什么：
        1. 读取 .cf.env 里的凭据（不回显任何密钥值）
        2. 打印当前登录身份 / 探测账号 ID
        3. 打印线上 Worker 当前状态（脚本名 + 现有 cron）
        4. npx wrangler deploy        覆盖线上代码，并按 wrangler.toml 应用 cron
        5. npx wrangler secret bulk   上传 GH_TOKEN / TRIGGER_TOKEN
        6. 访问 /health 做体检

    凭据准备：复制 .cf.env.example 为 .cf.env 并填写（该文件已被 gitignore）。
#>
param(
    [switch]$SkipDeploy,
    [switch]$SkipSecrets,
    [string]$WorkerUrl = 'https://lianhua.573183877.workers.dev'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$envFile = Join-Path $root '.cf.env'
if (-not (Test-Path $envFile)) {
    Write-Error "缺少 .cf.env —— 先复制 .cf.env.example 并填入 CLOUDFLARE_API_TOKEN / GH_TOKEN"
}

# ── 载入凭据到当前进程环境（只写 env:，不打印） ─────────────────────
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#') -and $line -match '^([^=]+)=(.*)$') {
        Set-Item -Path ('env:' + $matches[1].Trim()) -Value $matches[2].Trim()
    }
}

# ── TRIGGER_TOKEN 缺失则自动生成并写回 .cf.env ───────────────────────
if (-not $env:TRIGGER_TOKEN) {
    $random = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N').Substring(0, 8)
    $existing = Get-Content $envFile
    $found = $false
    $updated = foreach ($line in $existing) {
        if ($line -match '^\s*TRIGGER_TOKEN\s*=') {
            $found = $true
            "TRIGGER_TOKEN=$random"
        } else {
            $line
        }
    }
    if (-not $found) { $updated += "TRIGGER_TOKEN=$random" }
    Set-Content -Path $envFile -Value $updated -Encoding utf8
    Set-Item -Path 'env:TRIGGER_TOKEN' -Value $random
    Write-Host "[i] 已自动生成 TRIGGER_TOKEN 并写入 .cf.env"
}

$headers = @{
    'Authorization' = "Bearer $env:CLOUDFLARE_API_TOKEN"
    'Content-Type'  = 'application/json'
}
$api = 'https://api.cloudflare.com/client/v4'

if (-not $env:CLOUDFLARE_API_TOKEN) {
    Write-Error '.cf.env 里缺少 CLOUDFLARE_API_TOKEN'
}

# ── 账号 ID：没配就探测并写回 ────────────────────────────────────────
if (-not $env:CLOUDFLARE_ACCOUNT_ID) {
    $accounts = (Invoke-RestMethod -Uri "$api/accounts" -Headers $headers).result
    if (-not $accounts -or $accounts.Count -eq 0) {
        Write-Error '该 Token 看不到任何账号，检查 Token 的账号权限'
    }
    $env:CLOUDFLARE_ACCOUNT_ID = $accounts[0].id
    Add-Content -Path $envFile -Value "CLOUDFLARE_ACCOUNT_ID=$($accounts[0].id)"
    Write-Host "[i] 已探测到账号 $($accounts[0].name) / $($accounts[0].id)"
}
$accountId = $env:CLOUDFLARE_ACCOUNT_ID
$scriptName = ([regex]::Match((Get-Content (Join-Path $root 'wrangler.toml') -Raw -Encoding UTF8), '(?m)^name\s*=\s*"([^"]+)"')).Groups[1].Value
if (-not $scriptName) { Write-Error "无法从 wrangler.toml 解析 name" }

Write-Host '== 登录身份 =='
npx wrangler whoami

# ── 线上现状（部署前留底，方便对比/回滚） ─────────────────────────────
Write-Host "`n== 线上 Worker 现状 =="
try {
    $scripts = (Invoke-RestMethod -Uri "$api/accounts/$accountId/workers/scripts" -Headers $headers).result
    $names = @($scripts | ForEach-Object { $_.id })
    Write-Host "账号下的 Worker: $($names -join ', ')"
    if ($names -notcontains $scriptName) {
        Write-Warning "注意：账号里没有名为 '$scriptName' 的 Worker（线上可能是别的名字）"
    }
} catch {
    Write-Warning "读取脚本列表失败: $($_.Exception.Message)"
}

try {
    $sched = (Invoke-RestMethod -Uri "$api/accounts/$accountId/workers/scripts/$scriptName/schedules" -Headers $headers).result
    if ($sched -and @($sched).Count -gt 0) {
        Write-Host "现有 cron: $(@($sched) | ForEach-Object { $_.cron })"
    } else {
        Write-Host '现有 cron: （无）'
    }
} catch {
    Write-Warning "读取 cron 失败: $($_.Exception.Message)"
}

# ── 部署代码 ─────────────────────────────────────────────────────────
if (-not $SkipDeploy) {
    Write-Host "`n== 部署代码（会覆盖线上 Worker，并按 wrangler.toml 应用 cron） =="
    npx wrangler deploy
    if ($LASTEXITCODE -ne 0) { throw 'wrangler deploy 失败' }
}

# ── 上传 secrets ─────────────────────────────────────────────────────
if (-not $SkipSecrets) {
    if (-not $env:GH_TOKEN) { throw '.cf.env 里缺少 GH_TOKEN' }
    Write-Host "`n== 上传 secrets =="
    $secretsPath = Join-Path $root 'secrets.json'
    @{ GH_TOKEN = $env:GH_TOKEN; TRIGGER_TOKEN = $env:TRIGGER_TOKEN } |
        ConvertTo-Json | Set-Content -Path $secretsPath -Encoding utf8
    try {
        npx wrangler secret bulk $secretsPath
        if ($LASTEXITCODE -ne 0) { throw 'wrangler secret bulk 失败' }
    } finally {
        Remove-Item $secretsPath -ErrorAction SilentlyContinue
    }
    npx wrangler secret list
}

# ── 体检 ─────────────────────────────────────────────────────────────
Write-Host "`n== /health 体检 =="
try {
    $health = Invoke-RestMethod -Uri "$WorkerUrl/health"
    $health | ConvertTo-Json -Depth 6
} catch {
    Write-Warning "访问 /health 失败: $($_.Exception.Message)"
}

Write-Host "`n下一步：把 /health 的输出贴给助手；也可以手动验证 →"
Write-Host "  curl -X POST `"$WorkerUrl/trigger?token=<TRIGGER_TOKEN>`""
