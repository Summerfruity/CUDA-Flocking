param(
    [string]$Configuration = "Release",
    [int[]]$BlockSizes = @(32, 64, 128, 256, 512),
    [int]$Objects = 50000,
    [int]$Warmup = 30,
    [int]$Samples = 7,
    [int]$Steps = 30,
    [string]$Output = "benchmarks/blocksize_results_clean.csv"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$rows = [System.Collections.Generic.List[object]]::new()

foreach ($blockSize in $BlockSizes) {
    $buildDir = Join-Path $repoRoot ("build_bs{0}" -f $blockSize)
    $definition = "-DFLOCKING_BLOCK_SIZE={0}" -f $blockSize
    cmake -S $repoRoot -B $buildDir -G "Visual Studio 17 2022" -A x64 $definition
    cmake --build $buildDir --config $Configuration --target cis5650_benchmark --parallel 2

    $executable = Join-Path $buildDir ("bin/{0}/cis5650_benchmark.exe" -f $Configuration)
    $rawCsv = Join-Path $repoRoot ("benchmarks/block_{0}.csv" -f $blockSize)
    & $executable --mode coherent --objects $Objects --warmup $Warmup `
        --samples $Samples --steps $Steps --output $rawCsv
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark failed for block size $blockSize (exit $LASTEXITCODE)."
    }

    foreach ($row in Import-Csv $rawCsv) {
        $rows.Add([pscustomobject]@{
            block_size = $blockSize
            objects = [int]$row.objects
            sample = [int]$row.sample
            step_ms = [double]$row.step_ms
            steps_per_second = [double]$row.steps_per_second
            gpu = $row.gpu
        })
    }
}

$outputPath = Join-Path $repoRoot $Output
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null
$rows | Export-Csv $outputPath -NoTypeInformation
Write-Host "Wrote block-size results to $outputPath"
