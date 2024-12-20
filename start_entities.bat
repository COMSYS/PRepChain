@echo off
setlocal enabledelayedexpansion

:: Specify the path to the virtual environment's Python interpreter
set venv_path=%~dp0venv\Scripts\python.exe

:: Define the list of folders containing app.py scripts
set folders=reputationsystem\reputation_manager reputationsystem\pseudonym_manager reputationsystem\reputation_engine reputationsystem\votee reputationsystem\verification_engine reputationsystem\key_manager

:: Loop through each folder and start the app.py script in a separate terminal window
for %%f in (%folders%) do (
    start cmd /k "cd /d %~dp0%%f && !venv_path! app.py"
)

:: Optional: Wait for the user to press a key before closing the batch script
pause
