#!/bin/bash
# Reverse of archive_exps_nvme_to_hdd.sh: pull a directory back from the HDD to
# NVMe and remove the symlink.
#
# Why this exists: /mnt/HDD is exFAT with a 1 MiB cluster, so a directory of
# ~369 KB NPZ files costs about 2.8x its apparent size there -- 48G of NPZ
# caches occupied 136G. Bulk checkpoint dirs (a few large files each) have no
# such penalty and stay on the HDD.
set -eo pipefail

SRC_ROOT="/mnt/HDD/kojiek/meanaudio_exps"
DST_ROOT="/home/kojiek/exps_nvme"
LOG="/home/kojiek/logs/restore_exps_hdd_to_nvme.log"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

for name in "$@"; do
  LINK="$DST_ROOT/$name"
  SRC="$SRC_ROOT/$name"

  if [ ! -L "$LINK" ]; then log "[skip] $name is not a symlink"; continue; fi
  if [ ! -d "$SRC" ]; then log "[FAIL] $name missing on HDD"; continue; fi

  N_SRC=$(find "$SRC" -type f | wc -l)
  A_SRC=$(du -sb --apparent-size "$SRC" | cut -f1)
  log "[copy] $name  files=$N_SRC apparent=$A_SRC"

  TMP="$DST_ROOT/.restore_$name"
  rm -rf "$TMP"
  nice -n 10 ionice -c2 -n7 cp -r "$SRC" "$TMP"

  N_DST=$(find "$TMP" -type f | wc -l)
  A_DST=$(du -sb --apparent-size "$TMP" | cut -f1)
  if [ "$N_DST" -ne "$N_SRC" ] || [ "$A_DST" -ne "$A_SRC" ]; then
    log "[FAIL] $name verify mismatch (files $N_DST/$N_SRC bytes $A_DST/$A_SRC); leaving HDD copy"
    rm -rf "$TMP"; continue
  fi

  rm -f "$LINK"
  mv "$TMP" "$LINK"
  rm -rf "$SRC"
  log "[done] $name restored to NVMe"
done

log "[DONE] NVMe free: $(df -h /home/kojiek | tail -1 | awk '{print $4}')  HDD free: $(df -h /mnt/HDD | tail -1 | awk '{print $4}')"
