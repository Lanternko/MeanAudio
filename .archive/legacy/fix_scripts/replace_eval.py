#!/usr/bin/env python3
"""完整替換 eval 函數，避免 av_bench 依賴"""

with open('meanaudio/runner_flowmatching.py', 'r') as f:
    content = f.read()

# 註釋掉 av_bench imports
content = content.replace('from av_bench.evaluate import evaluate', '# from av_bench.evaluate import evaluate')
content = content.replace('from av_bench.extract import extract', '# from av_bench.extract import extract')

# 找到並替換整個 eval 函數
# 使用簡單的佔位符替換
old_eval = '''    @torch.inference_mode()
    def eval(self, audio_dir: Path, it: int, data_cfg: DictConfig) -> dict[str, float]:
        with torch.amp.autocast('cuda', enabled=False):
            if local_rank == 0:
                extract(audio_path=audio_dir,
                        output_path=audio_dir / 'cache',
                        device='cuda',
                        batch_size=16,  # btz=16: avoid OOM
                        num_workers=4,
                        skip_video_related=True,  # avoid extracting video related features 
                        audio_length=10) 
                output_metrics = evaluate(gt_audio_cache=Path(data_cfg.gt_cache),
                                          skip_video_related=True, 
                                          pred_audio_cache=audio_dir / 'cache')'''

new_eval = '''    @torch.inference_mode()
    def eval(self, audio_dir: Path, it: int, data_cfg: DictConfig) -> dict[str, float]:
        with torch.amp.autocast('cuda', enabled=False):
            output_metrics = {}  # Evaluation disabled for Phase 1
            if local_rank == 0:
                # Evaluation code disabled (av_bench not available)
                # extract(...) and evaluate(...) calls removed'''

content = content.replace(old_eval, new_eval)

# 寫回檔案
with open('meanaudio/runner_flowmatching.py', 'w') as f:
    f.write(content)

print("✅ eval 函數已完整替換")
