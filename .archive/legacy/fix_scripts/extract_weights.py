import torch

ckpt_path = './exps/phase3_stage1_hard_10k/phase3_stage1_hard_10k_ckpt_last.pth'
out_path = './exps/phase3_stage1_hard_10k/stage1_clean_weights.pth'

# 讀取完整 Checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu')

# 剝開外層，只提取 'model' 字典
sd = ckpt.get('model', ckpt)

new_sd = {}
for k, v in sd.items():
    # 清除可能造成干擾的 'model.' 或 'module.' 前綴
    clean_k = k.replace('module.', '').replace('model.', '')
    new_sd[clean_k] = v

# 預防性處理：將 t_embed 的權重複製一份給 r_embed
for k in list(new_sd.keys()):
    if 't_embed' in k:
        r_k = k.replace('t_embed', 'r_embed')
        new_sd[r_k] = new_sd[k].clone()

# 儲存純淨版權重
torch.save(new_sd, out_path)
print(f"✅ 純淨版權重已成功儲存至: {out_path}")
