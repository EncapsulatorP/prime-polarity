#!/usr/bin/env bash
set -euo pipefail

# Quick demo after `pip install .`
prime-polarity --start 100000 --end 120000 --windows 3 --window-size 5000
