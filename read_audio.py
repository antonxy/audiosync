from scipy.io import wavfile
import os
import subprocess
from tempfile import NamedTemporaryFile


def from_file(path):
    '''
    Converts file to wav using ffmpeg and reads the wav. Returns sample rate, audio data
    '''

    if not os.path.exists(path):
        raise ValueError('File not found')

    if not os.path.isfile(path):
        raise ValueError('Path is no file')

    if not os.access(path, os.R_OK):
        raise IOError('File is not readable')

    output = NamedTemporaryFile(mode="rb", delete=False)
    ffmpeg_call = ['ffmpeg',
                   '-y',  # always overwrite existing files
                   "-i", os.path.abspath(path),  # input_file options (filename last)
                   "-vn",  # Drop any video streams if there are any
                   "-ac", "1",  # Convert audio to mono
                   "-f", "wav",  # output options (filename last)
                   output.name
                   ]

    subprocess.call(ffmpeg_call, stderr=open(os.devnull))

    sr, data = wavfile.read(output.name)

    output.close()
    os.remove(output.name)

    return sr, data