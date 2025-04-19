#!/bin/bash

# Directory to be linted (defaults to the current directory)
DIRECTORY=${1:-$(pwd)}

# Check if the directory is either 'src' or 'test'
if [[ "$DIRECTORY" != "model_hub" && "$DIRECTORY" != "test" ]]; then
    echo "Error: The directory must be either 'src' or 'test'."
    exit 1
fi

# Check for --quiet flag
QUIET=false
if [[ "$2" == "--quiet" ]]; then
    QUIET=true
fi

# Ensure the directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi

# Define quiet mode arguments
ISORT_ARGS=""
BLACK_ARGS=""
PYLINT_ARGS="--jobs=4"
MYPY_ARGS=""

if $QUIET; then
    ISORT_ARGS="--quiet"
    BLACK_ARGS="--quiet"
    PYLINT_ARGS="$PYLINT_ARGS --score=n"
fi

# Run isort, black, and pylint on all Python files
echo "Running isort..."
isort $ISORT_ARGS "$DIRECTORY"

echo "Running black..."
black $BLACK_ARGS "$DIRECTORY"

echo "Running pylint..."
find "$DIRECTORY" -name "*.py" | xargs pylint $PYLINT_ARGS


echo "Running mypy..."
mypy $MYPY_ARGS "$DIRECTORY"