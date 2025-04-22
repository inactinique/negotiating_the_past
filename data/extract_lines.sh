#!/bin/bash

# This script extracts 500,000 random lines from a large CSV file
# Usage: ./extract_lines.sh input.csv output.csv

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
 echo "Usage: $0 input.csv output.csv"
 exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE=$2

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
 echo "Error: Input file '$INPUT_FILE' not found."
 exit 1
fi

# Extract header (first line)
head -n 1 "$INPUT_FILE" > "$OUTPUT_FILE"

# Get total number of lines (excluding header)
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
TOTAL_LINES=$((TOTAL_LINES - 1))

# Extract 100,000 random lines (excluding header)
tail -n +2 "$INPUT_FILE" | shuf -n 100000 >> "$OUTPUT_FILE"

echo "Successfully extracted 500,000 lines from $INPUT_FILE to $OUTPUT_FILE"