@echo off
set "FLUTTER_BIN=D:\vscode\fluttter\flutter\bin"
echo Adding Flutter to PATH for this session...
set "PATH=%PATH%;%FLUTTER_BIN%"

echo Checking Flutter version...
call flutter --version

echo Enabling Windows Desktop support...
call flutter config --enable-windows-desktop

cd mobile_app
echo Starting Flutter App...
echo If multiple devices are available, you may need to select one.
echo Running on Windows by default if no mobile device found...
call flutter run -d windows
pause
