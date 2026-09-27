@echo off
rem Protocol B, FT-Transformer, limited scope (tiers: base + 4p noise, then
rem Gaussianisation + full rotation, then remaining noise doses).
rem Scheduled via Task Scheduler task "pub2_h13_ftt". Stop: close the window
rem "h13-ftt", then  .venv\Scripts\python.exe src\h13_retuned.py --clear-locks
cd /d D:\pub2\src
..\.venv\Scripts\python.exe h13_retuned.py --clear-locks
start "h13-ftt" /min cmd /c "..\.venv\Scripts\python.exe -W ignore h13_retuned.py --families ftt --kinds base noise_features gaussianise rotate_full >> ..\logs\h13_ftt.log 2>&1"
