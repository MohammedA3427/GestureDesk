"""Build source/model/docs bundle without private data or machine-specific venv."""
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parent.parent
output=root.parent/'GestureDesk.zip'
excluded={'.venv','.git','__pycache__','data','work'}
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if not path.is_file() or any(part in excluded for part in relative.parts):continue
        if path.suffix in ('.pyc','.npz'):continue
        if relative.parts[0]=='reports' and path.name not in {'dashboard.png','landmarks-preview.png','drawing-preview.png','collection-preview.png'}:continue
        z.write(path,Path('GestureDesk')/relative)
print(output)
