import sys
import queue
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
assert np


q = queue.Queue()


def callback(indata, frames, time, status):
    """
    This is called from a separate thread for each audio block
    """
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


# Unique file name for every recording
filename = tempfile.mktemp(prefix='untitled_', suffix='.wav', dir='')
# Make sure the file is open before recording anything
with sf.SoundFile(filename, mode='x', samplerate=48000, channels=2) as file:
    with sd.InputStream(samplerate=48000, channels=2, callback=callback):
        print('#' * 80)
        print('press Ctrl+C to stop the recording')
        print('#' * 80)
        while True:
            file.write(q.get())
