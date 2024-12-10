set CONDAPATH=C:\Users\Jules\miniconda3\
set ENVNAME=SymbolScribe

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%
cd C:\Users\Jules\Documents\SymbolScribe\application

pyinstaller --noconfirm --onefile --windowed --add-data "C:/Users/Jules/miniconda3/envs/SymbolScribe/lib/site-packages/customtkinter;customtkinter/" --add-data "symbols/*.png;symbols" --add-data "SymbolCNN.onnx;." --add-data "icon.ico;." -i icon.ico .\SymbolScribe.py
pause