#!/bin/bash
echo "🎯 整合 MeanAudio 研究成果到 music_research_core"

# 創建目錄
mkdir -p ~/music_research_core/meanaudio_evaluation

# 複製所有測試腳本
echo "📋 複製測試腳本..."
cp test_rtf_baseline.py ~/music_research_core/meanaudio_evaluation/
cp test_music_only.py ~/music_research_core/meanaudio_evaluation/
cp test_bagpipes_bias.py ~/music_research_core/meanaudio_evaluation/

# 複製結果檔案
echo "📊 複製結果檔案..."
cp -r rtf_test_results ~/music_research_core/meanaudio_evaluation/
cp -r music_only_test ~/music_research_core/meanaudio_evaluation/
cp -r bagpipes_bias_test ~/music_research_core/meanaudio_evaluation/

# 複製完整報告
cp COMPLETE_EVALUATION_REPORT.md ~/music_research_core/meanaudio_evaluation/

# 創建 README
cat > ~/music_research_core/meanaudio_evaluation/README.md << 'DOC'
# MeanAudio 評估框架

## 快速開始
```bash
cd ~/music_research_core/meanaudio_evaluation

# RTF 基準測試
python test_rtf_baseline.py

# 音樂語義評估
python test_music_only.py

# LP-MusicCaps 偏見調查
python test_bagpipes_bias.py
```

## 主要發現

1. **RTF 性能**: 1.92（比論文慢 150 倍）
2. **語義保真度**: 0.42（音樂類），0.16（音效類）
3. **評估限制**: LP-MusicCaps 有 33.3% bagpipes 誤判率

詳見：[COMPLETE_EVALUATION_REPORT.md](COMPLETE_EVALUATION_REPORT.md)

## 結論

MeanAudio 技術可行但不建議作為主要研究方向。
建議專注於 DAC 音樂編解碼器研究。

## 環境需求

- RTX 5090（或 RTX 4090/5080）
- PyTorch 2.11+ (CUDA 12.8)
- LP-MusicCaps 模型
DOC

echo "✅ 整合完成！"
echo ""
echo "📁 檔案位置: ~/music_research_core/meanaudio_evaluation/"
echo "📄 完整報告: ~/music_research_core/meanaudio_evaluation/COMPLETE_EVALUATION_REPORT.md"
echo ""
ls -lh ~/music_research_core/meanaudio_evaluation/
