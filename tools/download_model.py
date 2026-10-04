"""Explicit one-time model download; live demo never needs the network."""
from pathlib import Path
import hashlib
import os
import urllib.request

URL='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
SHA256='fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
ROOT=Path(__file__).resolve().parent.parent

def main():
    target=ROOT/'models/hand_landmarker.task'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=SHA256:
            raise RuntimeError('Existing model checksum differs; move it aside and download again')
        print('Already available:',target)
        return
    temp=target.with_suffix('.download')
    try:
        with urllib.request.urlopen(URL,timeout=60) as response, temp.open('wb') as f:
            while block:=response.read(1024*1024):f.write(block)
        if hashlib.sha256(temp.read_bytes()).hexdigest()!=SHA256:
            raise RuntimeError('Model checksum mismatch')
        os.replace(temp,target)
        print('Downloaded official model. SHA256:',hashlib.sha256(target.read_bytes()).hexdigest())
    finally:
        temp.unlink(missing_ok=True)

if __name__=='__main__':main()
