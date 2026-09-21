param(
    [string]$BuildDir = "build",
    [string]$Configuration = "Release",
    [string]$Output = "benchmarks/results.csv",
    [int[]]$Objects = @(1000, 5000, 10000, 25000, 50000),
    [string[]]$Modes = @("naive", "scattered", "coherent"),
    [int]$Warmup = 20,
    [int]$Samples = 10,
    [int]$Steps = 20
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$benchmark = Join-Path $repoRoot "$BuildDir/bin/$Configuration/cis5650_benchmark.exe"
if (-not (Test-Path -LiteralPath $benchmark)) {
    throw "Benchmark executable not found at $benchmark. Build the cis5650_benchmark target first."
}

$outputPath = Join-Path $repoRoot $Output
$outputParent = Split-Path -Parent $outputPath
if ($outputParent) {
    New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
}
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}

$first = $true
foreach ($mode in $Modes) {
    foreach ($objectCount in $Objects) {
        $temp = Join-Path ([System.IO.Path]::GetTempPath()) ("boids_{0}_{1}_{2}.csv" -f $mode, $objectCount, $PID)
        try {
            & $benchmark --mode $mode --objects $objectCount --warmup $Warmup `
                --samples $Samples --steps $Steps --output $temp
            if ($LASTEXITCODE -ne 0) {
                throw "Benchmark failed for mode=$mode objects=$objectCount (exit $LASTEXITCODE)."
            }
            if ($first) {
                Get-Content -LiteralPath $temp | Set-Content -LiteralPath $outputPath
                $first = $false
            } else {
                Get-Content -LiteralPath $temp | Select-Object -Skip 1 |
                    Add-Content -LiteralPath $outputPath
            }
        } finally {
            Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "Wrote benchmark results to $outputPath"
