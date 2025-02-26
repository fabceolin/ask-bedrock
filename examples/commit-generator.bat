@echo off
REM Capture the JSON output from git diff processed by jq
for /f "delims=" %%i in ('git diff ^| jq -R --slurp "{ \"output\": . }"') do (
    set "json=%%i"
)
REM Pass the JSON string as a single argument to ask-bedrock
ask-bedrock prompt -p CommitGenerator "%json%"

