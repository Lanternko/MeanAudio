#!/bin/bash
# Archive closed-line experiment dirs from NVMe (exps_nvme) to the HDD, leaving a
# symlink behind at the original path.
#
# Why a symlink and not a plain move: dozens of scripts under ~/research and
# ~/MeanAudio/scripts reference these directories by absolute path (CLAUDE.md
# calls this out explicitly). The symlink keeps every one of those paths
# resolvable, which is the same pattern the phase4-phase8 dirs already use
# (exps_nvme/phase8_v2_stage1_400000 -> /mnt/HDD/kojiek/meanaudio_exps/...).
#
# /mnt/HDD is exFAT: no POSIX ownership, so `rsync -a` fails per-file with
# EPERM (memory reference_mnt_hdd_is_exfat.md). Plain cp -r only.
set -eo pipefail

SRC_ROOT="/home/kojiek/exps_nvme"
DST_ROOT="/mnt/HDD/kojiek/meanaudio_exps"
LOG="/home/kojiek/logs/archive_exps_nvme_to_hdd.log"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

[ -d "$DST_ROOT" ] || { log "[FAIL] $DST_ROOT missing"; exit 2; }

for name in "$@"; do
  SRC="$SRC_ROOT/$name"
  DST="$DST_ROOT/$name"

  if [ -L "$SRC" ]; then log "[skip] $name already a symlink"; continue; fi
  if [ ! -d "$SRC" ]; then log "[skip] $name not a directory"; continue; fi

  # Refuse anything a live process is holding open.
  if lsof +D "$SRC" >/dev/null 2>&1; then
    if lsof +D "$SRC" 2>/dev/null | awk 'NR>1 && $4 !~ /cwd/ {found=1} END{exit !found}'; then
      log "[FAIL] $name has open files; skipping"; continue
    fi
  fi

  N_SRC=$(find "$SRC" -type f | wc -l)
  B_SRC=$(du -sb "$SRC" | cut -f1)
  log "[copy] $name  files=$N_SRC bytes=$B_SRC"

  rm -rf "$DST"
  nice -n 10 ionice -c2 -n7 cp -r "$SRC" "$DST"

  N_DST=$(find "$DST" -type f | wc -l)
  B_DST=$(du -sb "$DST" | cut -f1)
  if [ "$N_DST" -ne "$N_SRC" ]; then
    log "[FAIL] $name file count $N_DST != $N_SRC; leaving NVMe copy intact"; continue
  fi
  # exFAT rounds allocation, so compare apparent size, not du blocks.
  A_SRC=$(du -sb --apparent-size "$SRC" | cut -f1)
  A_DST=$(du -sb --apparent-size "$DST" | cut -f1)
  if [ "$A_DST" -ne "$A_SRC" ]; then
    log "[FAIL] $name apparent bytes $A_DST != $A_SRC; leaving NVMe copy intact"; continue
  fi

  rm -rf "$SRC"
  ln -s "$DST" "$SRC"
  log "[done] $name -> $DST (freed $((B_SRC/1000000000))G)"
done

log "[DONE] remaining NVMe free: $(df -h /home/kojiek | tail -1 | awk '{print $4}')"
