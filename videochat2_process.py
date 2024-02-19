import os
import re
import subprocess

input_dir = "/home/wangyu/work/VideoChat2/data/internvid"
output_dir = "/home/wangyu/work/VideoChat2/data/internvid-10s"
os.makedirs(output_dir, exist_ok=True)

def extract_timestamp(filename):
    """ Extract the timestamp from the filename in the format 'LLU5X98aozs_648.258.mp4' """
    match = re.search(r'_(\d+).(\d+).mp4$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        raise ValueError("Invalid filename format")

def prune_video(filename, start_seconds, start_milliseconds, duration=10):
    """ Prune the video to a clip of specified duration starting from the given timestamp """
    output_filename = f"{output_dir}/{filename}"
    start_time = f"{start_seconds}.{start_milliseconds}"
    command = [
        'ffmpeg', '-ss', start_time, '-i', os.path.join(input_dir, filename), '-t', str(duration), 
        '-c', 'copy', output_filename
    ]
    subprocess.run(command)
    print(f"Pruned video saved as {output_filename}")

for file in os.listdir(input_dir):

    # Extract the timestamp
    start_seconds, start_milliseconds = extract_timestamp(file)

    # Prune the video
    prune_video(file, start_seconds, start_milliseconds)



