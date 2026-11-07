@echo off
cd /d D:\face_expression_classification_Comparison-main
set PYTHONUNBUFFERED=1
python cnn.py > results\cnn.log 2>&1
echo DONE >> results\cnn.log
