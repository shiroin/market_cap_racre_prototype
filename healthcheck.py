import os
import shutil
import sys

print(f"python={sys.version.split()[0]}")
print(f"port={os.getenv('PORT', '10000')}")
print(f"ffmpeg={shutil.which('ffmpeg') or 'missing'}")
try:
    import streamlit
    print(f"streamlit={streamlit.__version__}")
except Exception as exc:
    print(f"streamlit_import_error={exc}")
    raise
