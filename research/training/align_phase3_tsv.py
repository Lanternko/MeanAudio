import pandas as pd, os, glob
base_dir = "/home/kojiek/research_dev/meanaudio_training/phase3_hard_filtering"
tsv_path = os.path.join(base_dir, "phase3_hard_filtering.tsv")
df_orig = pd.read_csv(tsv_path, sep='\t')
captions_dict = {str(row['id']).split('_')[0]: row['caption'] for _, row in df_orig.iterrows()}
df_part = pd.read_csv(os.path.join(base_dir, "partition.tsv"), sep='\t')
npz_files = glob.glob(os.path.join(base_dir, "npz", "*.npz"))
valid_ids = sorted([int(os.path.basename(f).replace('.npz', '')) for f in npz_files if os.path.basename(f).replace('.npz', '').isdigit()])
new_rows = [{"id": i, "caption": captions_dict.get(os.path.basename(str(df_part.iloc[i][df_part.columns[0]])).replace(".wav", ""), "Unknown")} for i in valid_ids]
pd.DataFrame(new_rows).to_csv(tsv_path, sep='\t', index=False)
