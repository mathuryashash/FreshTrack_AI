@echo off
set /p image_path="Enter path to image (or press Enter for sample): "
if "%image_path%"=="" set image_path="data/New_Fruits/Bitter_Gourd/IMG_20200822_223831.jpg_0_1039.jpg"

echo Running prediction on %image_path%...
python scripts/predict_image.py %image_path%
pause
