#!/bin/bash

# Usage: ./clean_notebook_metadata.sh notebook.ipynb

if [ -z "$1" ]; then
  echo "Usage: $0 notebook.ipynb"
  exit 1
fi

NOTEBOOK="$1"

tmpfile=$(mktemp)
jq --indent 1 \
    '
    (.cells[] | select(has("outputs")) | .outputs) = []
    | (.cells[] | select(has("execution_count")) | .execution_count) = null
    | .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}}
    | .cells[].metadata = {}
    ' "$NOTEBOOK" > "$tmpfile" && mv "$tmpfile" "$NOTEBOOK"
