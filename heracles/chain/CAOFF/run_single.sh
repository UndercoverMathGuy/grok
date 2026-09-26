#!/bin/bash
cd "$(dirname "$0")"
for args in "23 1 2 1 0" "23 1 3 1 0" "23 1 4 1 0" "23 1 3 2 0" "23 1 3 3 0" "59 1 3 1 0" "59 1 5 1 0" "59 1 5 2 0" "59 1 8 1 0" "23 1 4 1 1 rand" "59 1 5 1 1 rand"; do
  uv run --with numpy --with torch python single_offset.py $args >> single.log 2>&1
done
echo DONE >> single.log
