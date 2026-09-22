param(
    [string]$OutBase = "bin/_ab2",
    [string]$Form = "top3_mean",
    [string]$RegionSet = "dyn_envelope",
    [int]$ChannelMode = 1
)

# NOTE: run this script from the repository root.  Never hard-code the repository
# path here: it contains non-ASCII characters and PowerShell 5.1 mangles them when
# reading a -File script.
$ErrorActionPreference = "Stop"

$configs = @(
    @{ name = "legacy_none"; feature = "legacy";  locator = "none" },
    @{ name = "compact_core"; feature = "compact"; locator = "core" }
)
$seeds = 42, 43, 44, 45, 46

foreach ($seed in $seeds) {
    foreach ($cfg in $configs) {
        $outDir = Join-Path $OutBase ("{0}_s{1}" -f $cfg.name, $seed)
        & python run_emd_experiments.py `
            --forms $Form `
            --region-set $RegionSet `
            --channel-mode $ChannelMode `
            --feature-set $cfg.feature `
            --locator-features $cfg.locator `
            --seed $seed `
            --output-dir $outDir | Out-Null
        Write-Host ("{0} seed {1} -> {2}" -f $cfg.name, $seed, $outDir)
    }
}
Write-Host "AB sweep finished"
