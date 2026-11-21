# Render a scene on a rented GPU, from this workstation.
#
#   pwsh tools/run_scene.ps1 -SshHost ssh8.vast.ai -SshPort 16096
#
# Steps:
#   1. packs clips, speaker turns and prompts into one tarball
#   2. uploads it plus tools/build_scene_pack.py and tools/scene_run.sh
#   3. builds one performance folder per shot on the instance
#   4. renders them (resumable) and assembles scene.mp4
#   5. downloads the mp4 locally
#
# The instance must run the wan22-s2v image with weights downloaded
# (`wan22-s2v download`) and SSH open. Nothing is rented by this script.
param(
    [Parameter(Mandatory = $true)][string]$SshHost,
    [Parameter(Mandatory = $true)][int]$SshPort,
    [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$Clips = "film/ltx-scene01/clips",
    [string]$Turns = "film/ltx-scene01/audio-scene01",
    [string]$Prompts = "scene01-pack",
    [string]$SceneSeconds = "116.1",
    [string]$Size = "480p",
    [int]$Steps = 20,
    [int]$InferFrames = 48,
    [int]$Fps = 24,
    [int]$Seed = 42,
    [string]$RemoteRoot = "/scene01",
    [string]$RemoteOut = "/scene01-out",
    [string]$OutFile = ".tmp/wan-results/scene.mp4"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $repo ".tmp/scene-pack"
$tar = Join-Path $repo ".tmp/scene-src.tgz"
$sshOpts = @("-i", $Key, "-p", "$SshPort", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "UserKnownHostsFile=.tmp/kh_scene", "-o", "BatchMode=yes")

Write-Host "== staging sources"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $stage "clips"),
    (Join-Path $stage "turns"), (Join-Path $stage "prompts") | Out-Null
Copy-Item (Join-Path $repo $Clips)/*.mp4 (Join-Path $stage "clips")
Copy-Item (Join-Path $repo $Turns)/*.wav (Join-Path $stage "turns")
Copy-Item (Join-Path $repo $Prompts)/* (Join-Path $stage "prompts") -Recurse
tar -czf $tar -C $stage .

Write-Host "== uploading to ${SshHost}:${SshPort}"
scp @sshOpts $tar "root@${SshHost}:/root/scene-src.tgz"
scp @sshOpts (Join-Path $repo "tools/build_scene_pack.py") "root@${SshHost}:/root/"
scp @sshOpts (Join-Path $repo "tools/scene_run.sh") "root@${SshHost}:/root/"

$remote = @"
set -euo pipefail
rm -rf /tmp/scene-src && mkdir -p /tmp/scene-src
tar -xzf /root/scene-src.tgz -C /tmp/scene-src
python3 /root/build_scene_pack.py --clips /tmp/scene-src/clips --turns /tmp/scene-src/turns --prompts /tmp/scene-src/prompts --timings /tmp/scene-src/prompts/manifest.csv --out $RemoteRoot --scene-seconds $SceneSeconds
cd /opt/wan-docker
SIZE=$Size STEPS=$Steps INFER_FRAMES=$InferFrames FPS=$Fps SEED=$Seed bash /root/scene_run.sh $RemoteRoot $RemoteOut
"@

Write-Host "== building and rendering (the long part)"
$remote | ssh @sshOpts "root@$SshHost" "bash -s"

Write-Host "== downloading result"
$localOut = Join-Path $repo $OutFile
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $localOut) | Out-Null
scp @sshOpts "root@${SshHost}:$RemoteOut/scene.mp4" $localOut
Write-Host "saved $localOut"
