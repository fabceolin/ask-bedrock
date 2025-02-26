# commit-generator.ps1

# Get the current script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Step 1: Generate JSON from git diff using jq.
# This converts the git diff output into a JSON object.
$jsonOutput = git diff | jq -R --slurp '{ "output": . }'

# Step 2: Save the JSON output to a file named "data.json" with absolute path
$dataFile = Join-Path $scriptDir "data.json"
Write-Host "Creating data file at: $dataFile"

# PowerShell before 6.0 does not support utf8NoBOM, so we need to check the version
$psVersion = $PSVersionTable.PSVersion.Major
if ($psVersion -ge 6) {
    # PowerShell Core (6.0+) supports utf8NoBOM
    $jsonOutput | Out-File -FilePath $dataFile -Encoding utf8NoBOM
} else {
    # For Windows PowerShell 5.1 and earlier, use UTF8 but be careful about BOM
    [System.IO.File]::WriteAllText($dataFile, $jsonOutput, [System.Text.Encoding]::UTF8)
}

# Verify file was created
if (Test-Path $dataFile) {
    Write-Host "Data file created successfully"
    # Show the file content for debugging
    Write-Host "File content:"
    Get-Content $dataFile | Select-Object -First 3
} else {
    Write-Host "Failed to create data file!"
    exit 1
}

# Step 3: Call ask-bedrock with the preset and the data file.
Write-Host "Running: ask-bedrock prompt -p CommitGenerator -d $dataFile"
ask-bedrock prompt -p CommitGenerator -d $dataFile
