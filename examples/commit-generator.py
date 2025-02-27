import subprocess
import json
import sys
import platform

def run_command(command):
    """Execute a shell command and return its output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def main():
    # Ensure Git is installed
    if run_command("git --version") is None:
        print("Git is not installed or not in the system PATH.", file=sys.stderr)
        sys.exit(1)

    # Get the Git diff output
    git_diff = run_command("git diff")

    # Convert to JSON format
    json_payload = json.dumps({"output": git_diff})

    # Construct the ask-bedrock command
    command = f'ask-bedrock prompt -p CommitGenerator "{json_payload}"'

    # Run the command
    output = run_command(command)

    # Print the output
    print(output)

if __name__ == "__main__":
    main()

