@echo off
cd /d D:\face_expression_classification_Comparison-main
set PYTHONUNBUFFERED=1
python run_all.py > results\console.log 2>&1
if exist results\qwen_finetune.json python export_gguf.py > results\export_gguf.log 2>&1
echo FINISHED >> results\console.log
