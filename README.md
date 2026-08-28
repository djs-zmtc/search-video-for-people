# Scripts to Process Security Video

## Run Order

1. Run `uv run python .\repair_clips.py` first, to fix any clip errors from the NVR dumps
2. Run `uv run python .\filter_footage.py` second, which will use the `repaired_videos` folder created by step #1

Any clips that the AI analysis detects with potential people in them will be placed in the `person_clips` folder.

