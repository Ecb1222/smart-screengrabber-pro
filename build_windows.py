#!/usr/bin/env python3
"""
Windows build script for Smart Screengrabber Pro
Run this on a Windows PC to create SmartScreengrabberPro.exe
"""

import subprocess
import sys
import os
import shutil

def clean_build():
    """Remove previous build artifacts"""
    dirs_to_clean = ['build', 'dist']
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"Removing {d}...")
            shutil.rmtree(d)
    
    spec_files = [f for f in os.listdir('.') if f.endswith('.spec')]
    for f in spec_files:
        print(f"Removing {f}...")
        os.remove(f)

def build_app():
    """Build the Windows app using PyInstaller"""
    print("\n" + "="*60)
    print("Building Smart Screengrabber Pro for Windows")
    print("="*60)
    print("\nThis will take 10-15 minutes...")
    print("Progress will be logged to build_windows.log\n")
    
    # Get MediaPipe modules path
    import mediapipe
    mediapipe_path = os.path.dirname(mediapipe.__file__)
    mediapipe_modules = os.path.join(mediapipe_path, 'modules')
    
    # Get OpenCLIP path for vocabulary files
    import open_clip
    open_clip_path = os.path.dirname(open_clip.__file__)
    
    # NOTE: Windows uses semicolon (;) instead of colon (:) for --add-data
    cmd = [
        sys.executable,
        '-m', 'PyInstaller',
        '--name=SmartScreengrabberPro',
        '--windowed',  # No console window (GUI app)
        '--onedir',  # Directory bundle (faster than --onefile)
        '--clean',
        '--noconfirm',
        # --add-data format on Windows: source;destination
        '--add-data=yolov8n.pt;.',  # Include YOLO model
        f'--add-data={mediapipe_modules};mediapipe/modules',  # MediaPipe data
        f'--add-data={open_clip_path}\\*.txt.gz;open_clip',  # OpenCLIP vocab
        # Exclude heavy unnecessary modules (keep sympy for OpenCLIP)
        '--exclude-module=tensorboard',
        '--exclude-module=notebook',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
        '--exclude-module=pytest',
        '--exclude-module=sphinx',
        '--exclude-module=jax',
        '--exclude-module=jaxlib',
        'grab_enhanced.py'
    ]
    
    # Run with output to both console and log file
    with open('build_windows.log', 'w') as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("Building", end='', flush=True)
        dot_counter = 0
        for line in process.stdout:
            log.write(line)
            log.flush()
            # Show progress dots
            if 'INFO:' in line:
                print('.', end='', flush=True)
                dot_counter += 1
                if dot_counter % 50 == 0:
                    print(f' ({dot_counter})', flush=True)
                    print("Building", end='', flush=True)
        
        print()  # New line
        process.wait()
        return process.returncode

def main():
    print("Smart Screengrabber Pro - Windows Build Script")
    print("=" * 60)
    
    # Check if running on Windows
    if sys.platform != 'win32':
        print("\n❌ ERROR: This script must be run on Windows!")
        print("You are currently on:", sys.platform)
        print("\nTo build for Windows:")
        print("1. Copy files to a Windows PC")
        print("2. Install Python and dependencies")
        print("3. Run this script on Windows")
        return 1
    
    # Step 1: Clean
    print("\n[1/2] Cleaning previous build artifacts...")
    clean_build()
    
    # Step 2: Build
    print("\n[2/2] Building Windows application...")
    returncode = build_app()
    
    if returncode == 0:
        print("\n" + "="*60)
        print("BUILD SUCCESSFUL!")
        print("="*60)
        print("\nYour Windows app is ready at:")
        print("  dist\\SmartScreengrabberPro\\SmartScreengrabberPro.exe")
        print("\nTo test:")
        print("  dist\\SmartScreengrabberPro\\SmartScreengrabberPro.exe")
        print("\nTo distribute:")
        print("  1. Right-click dist\\SmartScreengrabberPro folder")
        print("  2. Send to > Compressed (zipped) folder")
        print("  3. Share the ZIP file (~700 MB)")
        print("\nRecipients just:")
        print("  1. Unzip the folder")
        print("  2. Double-click SmartScreengrabberPro.exe")
        print("  3. No Python or dependencies needed!")
    else:
        print("\n" + "="*60)
        print("BUILD FAILED")
        print("="*60)
        print("\nCheck build_windows.log for details.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r Requirements.txt")
        print("  - PyInstaller not installed: pip install pyinstaller")
        print("  - Antivirus blocking: Add exception for build folder")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
