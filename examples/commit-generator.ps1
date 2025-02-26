# commit-generator.ps1

# Step 1: Generate JSON from git diff using jq.
# This converts the git diff output into a JSON object.
$jsonOutput = git diff | jq -R --slurp '{ "output": . }'

# Step 2: Save the JSON output to a file with full path
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataFile = Join-Path $scriptDir "data.json"

# PowerShell before 6.0 does not support utf8NoBOM, so check the version
$psVersion = $PSVersionTable.PSVersion.Major
if ($psVersion -ge 6) {
    # PowerShell Core (6.0+) supports utf8NoBOM
    $jsonOutput | Out-File -FilePath $dataFile -Encoding utf8NoBOM
} else {
    # For older PowerShell versions
    $jsonOutput | Out-File -FilePath $dataFile -Encoding utf8

    # Read the file as bytes, check for BOM, and rewrite if needed
    $bytes = [System.IO.File]::ReadAllBytes($dataFile)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 239 -and $bytes[1] -eq 187 -and $bytes[2] -eq 191) {
        # Remove BOM (first 3 bytes)
        $newBytes = $bytes[3..($bytes.Length-1)]
        [System.IO.File]::WriteAllBytes($dataFile, $newBytes)
    }
}

# Display the file location for debugging
Write-Host "Data file saved at: $dataFile"

# Step 3: Call ask-bedrock with the preset and the data file.
ask-bedrock prompt -p CommitGenerator -d $dataFile
