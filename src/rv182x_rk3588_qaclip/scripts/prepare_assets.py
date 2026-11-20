"""Restore the byte-identical QAClip vision weight from Git-sized chunks."""
import hashlib
import json
from pathlib import Path
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    model = Path(__file__).resolve().parents[1] / 'model'
    manifest = json.loads((model / 'qaclip-vision.parts.json').read_text())
    target = model / manifest['filename']
    if target.is_file() and target.stat().st_size == manifest['size'] and sha256(target) == manifest['sha256']:
        print('QAClip vision weight already verified')
        return
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=model, prefix='.qaclip-vision-', delete=False) as output:
            temp = Path(output.name)
            full = hashlib.sha256()
            total = 0
            for entry in manifest['parts']:
                part = model / entry['filename']
                digest = hashlib.sha256()
                count = 0
                with part.open('rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(block)
                        full.update(block)
                        output.write(block)
                        count += len(block)
                if count != entry['size'] or digest.hexdigest() != entry['sha256']:
                    raise RuntimeError(f'Invalid model chunk: {part.name}')
                total += count
        if total != manifest['size'] or full.hexdigest() != manifest['sha256']:
            raise RuntimeError('Assembled QAClip vision weight does not match the tested model')
        temp.chmod(0o644)
        temp.replace(target)
        print('Restored and SHA256-verified QAClip vision weight')
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
