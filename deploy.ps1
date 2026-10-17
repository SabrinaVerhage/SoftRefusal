# Run from PowerShell: .\deploy.ps1 -Push
# Commit changes first. Use -Check to compare versions without deploying.
[CmdletBinding()]
param(
    [switch]$Push,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($Push -and $Check) { throw 'Choose either -Push or -Check.' }

# Prefer Windows Git and OpenSSH over unrelated MSYS installations on PATH.
$gitExe = Join-Path $env:ProgramFiles 'Git\cmd\git.exe'
if (-not (Test-Path -LiteralPath $gitExe)) {
    $gitExe = (Get-Command git.exe -ErrorAction Stop).Source
}
$sshExe = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$site = 'https://softrefusal.sabrinaverhage.com'

function Invoke-RepositoryGit {
    param([string[]]$GitArguments)
    $result = & $gitExe -C $PSScriptRoot @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($GitArguments -join ' ')" }
    return $result
}

function Get-PublishedCommit {
    $line = Invoke-RepositoryGit @('ls-remote', '--exit-code', 'origin', 'refs/heads/main')
    $commit = ($line -split '\s+')[0]
    if ($commit -notmatch '^[0-9a-f]{40}$') { throw 'Could not identify GitHub main.' }
    return $commit
}

$branch = Invoke-RepositoryGit @('branch', '--show-current')
$localCommit = Invoke-RepositoryGit @('rev-parse', 'HEAD')
$changes = Invoke-RepositoryGit @('status', '--porcelain')

if ($Check) {
    $published = Get-PublishedCommit
    $health = Invoke-RestMethod -Uri "$site/healthz" -TimeoutSec 20 -Headers @{'Cache-Control' = 'no-cache'}
    Write-Host "Local ($branch): $localCommit"
    Write-Host "GitHub main:     $published"
    Write-Host "Live VPS:        $($health.version) ($($health.status))"
    if ($changes) { Write-Host 'Local changes have not been committed.' }
    if ($health.version -eq $published -and $health.status -eq 'ok') {
        Write-Host 'The VPS is up to date with GitHub main.'
    } else {
        Write-Warning 'The VPS does not report a healthy deployment of GitHub main.'
    }
    return
}

if ($branch -ne 'main') { throw 'Switch to main before deploying. Other branches are not deployed by this script.' }
if ($changes) { throw 'Commit your changes first, then run .\deploy.ps1 -Push.' }
if (-not (Test-Path -LiteralPath $sshExe)) { throw 'Install Windows OpenSSH Client before deploying.' }

if ($Push) {
    Write-Host 'Pushing the committed main branch to GitHub...'
    Invoke-RepositoryGit @('push', 'origin', 'HEAD:refs/heads/main')
}
$published = Get-PublishedCommit
if ($localCommit -ne $published) {
    throw 'Local main and GitHub main differ. Pull the latest changes, or use -Push for your committed updates.'
}

Write-Host "Deploying $published to Greenhost..."
& $sshExe -o BatchMode=yes -o ConnectTimeout=15 root@213.108.108.24 "softrefusal-deploy $published"
if ($LASTEXITCODE -ne 0) {
    throw 'Deployment did not complete. Read the server message above; if visitors are recording, retry when idle. Any successful GitHub push remains saved.'
}

for ($attempt = 1; $attempt -le 3; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "$site/healthz" -TimeoutSec 20 -Headers @{'Cache-Control' = 'no-cache'}
        if ($health.status -ne 'ok' -or $health.version -ne $published) {
            throw 'The public site is not reporting the requested healthy version.'
        }
        Write-Host "Verified live: $site ($published)"
        return
    } catch {
        if ($attempt -eq 3) {
            throw "The server deployment finished, but public verification failed. Run .\deploy.ps1 -Check to inspect it. $($_.Exception.Message)"
        }
        Start-Sleep -Seconds 2
    }
}
