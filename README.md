audiosync
=========
Tool to sync audio and video automatically using a sync pattern which also contains scene, shot, take.
You can generate these patterns with this [app for android](https://github.com/antonxy/audiosync_androidapp).
This tool is only a quick hack for now and I only tested it on Linux.

It depends on:
- Python
- numpy
- scipy
- ffmpeg

It finds the sync points in the audio of all files (Camera needs some mic too), renames the files to scene-shot-take and generates EDL files for Lightworks (might work with other editors too)

You can run it like this:
`python console.py dir-with-all-video-files/ dir-with-all-audio-files/ edl-output-dir/ project_frames_per_second`