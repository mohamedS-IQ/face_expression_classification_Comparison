@echo off
cd /d D:\face_expression_classification_Comparison-main
set PYTHONUNBUFFERED=1
for %%s in (mlp cnn qwen_zeroshot qwen_finetune) do (
  echo ===== %%s start %date% %time% =====
  python %%s.py > results\%%s.log 2>&1
  echo ===== %%s exit %errorlevel% %date% %time% =====
)
echo ALL DONE
