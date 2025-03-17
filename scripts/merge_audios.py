import argparse
import os
import sys
import wave  # Add this import

from pydub import AudioSegment


def is_valid_wav(file_path):
    """Check if file is actually a WAV file with proper header"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            return True
    except:
        return False

def repair_and_load(file_path):
    """Attempt automatic repair for common header issues"""
    try:
        # Try standard load first
        return AudioSegment.from_wav(file_path)
    except:
        # Fallback to FFmpeg with header repair
        return AudioSegment.from_file(file_path, format="wav")


def stitch_wav_files(input_dir, output_file, silence_ms=0):
    """
    Combines multiple WAV files into a single WAV file.
    """
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Directory {input_dir} not found")
        return False

    # Get all WAV files in the directory
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return False

    # Sort the files to ensure consistent order
    wav_files.sort()
    print(f"Found {len(wav_files)} WAV files to stitch")

    # Initialize combined as None
    combined = None

    # Process all files
    for wav_file in wav_files:
        file_path = os.path.join(input_dir, wav_file)
        try:
            # Add format validation first
            if not is_valid_wav(file_path):
                print(f"⚠️ Invalid WAV header in {wav_file} - skipping")
                # continue

            # Try loading with both methods
            try:
                audio = AudioSegment.from_wav(file_path)  # First try native WAV
            except:
                audio = AudioSegment.from_file(file_path)  # Fallback to FFmpeg

    # for wav_file in wav_files:
    #     file_path = os.path.join(input_dir, wav_file)
    #     try:
    #         audio = AudioSegment.from_file(file_path, format="wav")
            print(f"Successfully loaded {wav_file}")

            if combined is None:
                combined = audio
            else:
                # Add silence between files if requested
                if silence_ms > 0:
                    combined += AudioSegment.silent(duration=silence_ms)
                combined += audio

        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")
            print(f"Skipping this file and continuing...")

    if combined is None:
        print("No audio files could be processed successfully")
        return False

    try:
        # Export the combined audio to a WAV file
        combined.export(output_file, format="wav")
        print(f"Successfully created stitched WAV file: {output_file}")
        return True
    except Exception as e:
        print(f"Error exporting final audio: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stitch multiple WAV files into a single WAV file')
    parser.add_argument('--input_dir', required=True, help='Directory containing WAV files')
    parser.add_argument('--output_file', required=True, help='Path to save the combined WAV file')
    parser.add_argument('--silence_ms', type=int, default=0, help='Milliseconds of silence to add between files')

    args = parser.parse_args()
    success = stitch_wav_files(args.input_dir, args.output_file, args.silence_ms)
    sys.exit(0 if success else 1)

