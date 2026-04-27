# Run DOPNet inference on all panorama images
# Source repo: D:\yuzu-windows-msvc\DOPNet
# Output raw JSON: D:\yuzu-windows-msvc\DOPNet\outputs\2_raw_dop_temp

cd D:\yuzu-windows-msvc\DOPNet

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

$inputDir = "D:\yuzu-windows-msvc\DOPNet\input_images"
$outputDir = "D:\yuzu-windows-msvc\DOPNet\outputs\2_raw_dop_temp"
$timeLog = "D:\yuzu-windows-msvc\DOPNet\outputs\2_raw_dop_temp\inference_time_dopnet.csv"

if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

if (!(Test-Path $timeLog)) {
    "image_name,inference_time_sec,status" | Set-Content $timeLog
}

$imgs = Get-ChildItem "$inputDir\*.png"
$total = $imgs.Count
$i = 0

foreach ($img in $imgs) {
    $i++
    $name = $img.BaseName
    $rawPath = Join-Path $outputDir "$name`_raw.json"

    if (Test-Path $rawPath) {
        Write-Host "[$i/$total] Skip existing: $name"
        continue
    }

    Write-Host "[$i/$total] Running: $name"

    $elapsed = Measure-Command {
        python inference.py `
          --cfg src/my_config/mp3d.yaml `
          --img_glob "$($img.FullName)" `
          --output_dir "$outputDir" `
          --post_processing manhattan `
          --device cpu
    }

    if (Test-Path $rawPath) {
        "$name,$([math]::Round($elapsed.TotalSeconds, 3)),ok" | Add-Content $timeLog
        Write-Host "Done: $name"
    } else {
        "$name,$([math]::Round($elapsed.TotalSeconds, 3)),failed" | Add-Content $timeLog
        Write-Host "Failed: $name"
    }

    [System.GC]::Collect()
}
