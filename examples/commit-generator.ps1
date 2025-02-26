# commit-generator.ps1

# Step 1: Generate JSON from git diff using jq.
# This converts the git diff output into a JSON object.
$jsonOutput = git diff | jq -R --slurp '{ "output": . }'

# Step 2: Save the JSON output to a file named "data.json".
$dataFile = "data.json"

# PowerShell before 6.0 does not support utf8NoBOM, so we need to check the version
$psVersion = $PSVersionTable.PSVersion.Major
if ($psVersion -ge 6) {
    # PowerShell Core (6.0+) supports utf8NoBOM
    $jsonOutput | Out-File -FilePath $dataFile -Encoding utf8NoBOM
} else {
    # For older PowerShell versions, we need to manually remove BOM
    # First, write to a temp file
    $tempFile = [System.IO.Path]::GetTempFileName()
    $jsonOutput | Out-File -FilePath $tempFile -Encoding utf8

    # Read the content as bytes, skip the BOM, and write to the final file
    $bytes = [System.IO.File]::ReadAllBytes($tempFile)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 239 -and $bytes[1] -eq 187 -and $bytes[2] -eq 191) {
        # Remove BOM (first 3 bytes)
        $bytes = $bytes[3..($bytes.Length-1)]
    }
    [System.IO.File]::WriteAllBytes($dataFile, $bytes)

    # Clean up
    Remove-Item $tempFile
}

# Step 3: Call ask-bedrock with the preset and the data file.
ask-bedrock prompt -p CommitGenerator -d $dataFile
