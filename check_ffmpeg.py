import os
from pathlib import Path

# The paths currently defined in your main.py
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
ffprobe_path = r"C:\ffmpeg\bin\ffprobe.exe"

print("=" * 50)
print("FFMPEG PATH DIAGNOSTICS")
print("=" * 50)

print(f"\n1. Checking ffprobe path: {ffprobe_path}")
if os.path.exists(ffprobe_path):
    print("   ✅ SUCCESS: File exists!")
else:
    print("   ❌ ERROR: File NOT found at this exact location.")

print(f"\n2. Checking ffmpeg path: {ffmpeg_path}")
if os.path.exists(ffmpeg_path):
    print("   ✅ SUCCESS: File exists!")
else:
    print("   ❌ ERROR: File NOT found at this exact location.")

# Let's inspect what is actually inside C:\ if it exists
base_dir = r"C:\ffmpeg"
print(f"\n3. Checking base directory structure: {base_dir}")
if os.path.exists(base_dir):
    print(f"   Contents of {base_dir}:")
    try:
        for item in os.listdir(base_dir):
            print(f"     - {item}")
    except Exception as e:
        print(f"     Error reading directory: {e}")
else:
    print("   ❌ The directory C:\\ffmpeg does not exist at all.")
    
    # Check if there is a similarly named folder instead
    print("\n4. Scanning C:\\ root for alternative folders...")
    try:
        c_items = os.listdir("C:/")
        matches = [item for item in c_items if "ffmpeg" in item.lower()]
        if matches:
            print("   Found these potential matches on C:\\:")
            for m in matches:
                print(f"     - C:\\{m}")
        else:
            print("   No folders containing 'ffmpeg' found directly on C:\\")
    except Exception as e:
        print(f"   Could not read C:\\ root: {e}")
print("=" * 50)