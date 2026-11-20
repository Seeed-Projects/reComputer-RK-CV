from pathlib import Path
r=Path(__file__).resolve().parent
files=sorted((r/'calibration').glob('*.jpg'))
(r/'calibration.txt').write_text('\n'.join(str(p.resolve()) for p in files)+'\n')
