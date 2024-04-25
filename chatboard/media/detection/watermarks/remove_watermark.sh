#!/usr/bin/env bash

# ./watermark.py "$tmpdir"

echo "Removing watermark in video..."
ffmpeg -hide_banner -loglevel warning -y -stats -i "$1" -acodec copy -vf "removelogo=$tmpdir/mask.png" "$output_file"

rm -rf "$tmpdir"

echo "Done"

exit 0
