from pathlib import Path
r=Path(__file__).resolve().parent
lines=[]
for line in (r/'calibration/image_pairs.txt').read_text().splitlines():
    lines.append(' '.join(str((r/'calibration'/name).resolve()) for name in line.split()))
(r/'calibration.txt').write_text('\n'.join(lines)+'\n')
