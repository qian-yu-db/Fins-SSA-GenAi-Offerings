#!/bin/bash

# Display usage information
function show_usage {
  echo "Usage: $0 [--base-dir=PATH] [--output-dir=PATH]"
  echo ""
  echo "Options:"
  echo "  --base-dir=PATH    Directory containing audio subdirectories (default: $BASE_DIR)"
  echo "  --output-dir=PATH  Directory to save merged audio files (default: $OUTPUT_DIR)"
  echo ""
  exit 1
}

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --base-dir=*)
      BASE_DIR="${arg#*=}"
      ;;
    --output-dir=*)
      OUTPUT_DIR="${arg#*=}"
      ;;
    --help)
      show_usage
      ;;
    *)
      echo "Unknown argument: $arg"
      show_usage
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Find all subdirectories in the base directory
for DIR in "$BASE_DIR"/*/ ; do
    # Extract directory name for output file naming
    DIR_NAME=$(basename "$DIR")

    # Skip if not a directory
    if [ ! -d "$DIR" ]; then
        continue
    fi

    echo "Processing directory: $DIR_NAME"

    # Define output file name
    OUTPUT_FILE="$OUTPUT_DIR/${DIR_NAME}.wav"

    # Run the merge_audios.py script
    python ./scripts/merge_audios.py --input_dir "$DIR" --output_file "$OUTPUT_FILE" --silence_ms 500

    # Check if merge was successful
    if [ $? -eq 0 ]; then
        echo "Successfully created $OUTPUT_FILE"
    else
        echo "Failed to process $DIR_NAME"
    fi

    echo "------------------------"
done

echo "Processing complete. Merged files are in the $OUTPUT_DIR directory."