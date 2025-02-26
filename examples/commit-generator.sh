ask-bedrock prompt -p CommitGenerator "$(git diff | jq -R --slurp '{ "output": . }')"
