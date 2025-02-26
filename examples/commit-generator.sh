ask-bedrock prompt -p CommitGenerator -j "$(git diff | jq -R --slurp '{ "output": . }')"
