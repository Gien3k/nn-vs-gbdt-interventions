@echo off
rem Starts the H13 workers (resumes: finished units are skipped).
rem Scheduled via Task Scheduler task "pub2_h13". Stop: close the windows
rem or run  taskkill /FI "WINDOWTITLE eq h13-*"  then
rem  .venv\Scripts\python.exe src\h13_retuned.py --clear-locks
cd /d D:\pub2\src
..\.venv\Scripts\python.exe h13_retuned.py --clear-locks
start "h13-mlp"    /min cmd /c "..\.venv\Scripts\python.exe -W ignore h13_retuned.py --families mlp    >> ..\logs\h13_mlp.log 2>&1"
start "h13-resnet" /min cmd /c "..\.venv\Scripts\python.exe -W ignore h13_retuned.py --families resnet >> ..\logs\h13_resnet.log 2>&1"
start "h13-xgb"    /min cmd /c "..\.venv\Scripts\python.exe -W ignore h13_retuned.py --families xgb    >> ..\logs\h13_xgb.log 2>&1"
start "h13-lgbm"   /min cmd /c "..\.venv\Scripts\python.exe -W ignore h13_retuned.py --families lgbm   >> ..\logs\h13_lgbm.log 2>&1"
