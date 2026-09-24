import glob
import os
import pandas as pd

npz_dir = "/home/kojiek/research_dev/meanaudio_training/phase3_baseline/npz"
tsv_path = "/home/kojiek/research_dev/meanaudio_training/phase3_baseline/phase3_baseline.tsv"

npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
valid_ids = sorted([
    int(os.path.basename(f).replace('.npz', '')) 
    for f in npz_files 
    if os.path.basename(f).replace('.npz', '').isdigit()
])

print(f"🔍 掃描到 {len(valid_ids)} 個實際存在的 NPZ 檔案。")
new_rows = [{"id": i, "caption": "Unknown"} for i in valid_ids]
pd.DataFrame(new_rows).to_csv(tsv_path, sep='\t', index=False)
