#!/usr/bin/env bash
set -euo pipefail

root_dir="${1:-data/external/raw/nflverse}"
mkdir -p "$root_dir"

curl -L --fail --silent --show-error \
  https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv \
  -o "$root_dir/games.csv"

for season in 2020 2021 2022 2023 2024
do
  curl -L --fail --silent --show-error \
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_${season}.csv" \
    -o "$root_dir/injuries_${season}.csv"
  curl -L --fail --silent --show-error \
    "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_${season}.csv" \
    -o "$root_dir/roster_weekly_${season}.csv"
  curl -L --fail --silent --show-error \
    "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_${season}.csv" \
    -o "$root_dir/snap_counts_${season}.csv"
done

printf '%s\n' "External NFL data saved to $root_dir"
