#!/bin/bash
# rent_and_launch.sh <box> <disk> <offer...>: rent_good.sh, then launch_box.sh on the box it keeps (2026-11-13 J4 split).
V=$(cd "$(dirname "$0")" && pwd)
box=$1; disk=$2; shift 2
out=$(MIN_MBPS=${MIN_MBPS:-50} bash "$V/rent_good.sh" "$box" "$disk" "$@" 2>&1); echo "$out"
id=$(echo "$out" | awk -v b="$box" '$1=="GOOD" && $2==b {print $3}' | tail -1)
[ -n "$id" ] || { echo "RENT_FAILED $box"; exit 1; }
bash "$V/launch_box.sh" "$id" "$box" && echo "LAUNCHED $box $id" || echo "LAUNCH_FAILED $box $id"
