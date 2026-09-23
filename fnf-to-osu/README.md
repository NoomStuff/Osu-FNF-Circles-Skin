# FNF note conversion

Run `python fnf-to-osu/convert_notes.py --install` from the skin directory.
The script reads the current note size, generates the assets in
`fnf-to-osu/convert`, then copies them into the skin. It uses `Notes.png`, the
supplied `Spashes.png`, and the official V-Slice `holdCoverBlue` atlas.

The cover source is [FunkinCrew's asset repository](https://github.com/FunkinCrew/funkin.assets/tree/main/shared/images).
V-Slice uses [separate start, loop, and end cover animations](https://github.com/FunkinCrew/Funkin/blob/main/source/funkin/play/notes/NoteHoldCover.hx).
osu!mania has only one shared, looping hold-light animation, so this skin uses
the four official loop frames. It also has one shared tap-light animation, so
the supplied splash is rendered in white for every lane. The lane notes keep
their individual colors. osu!mania's pressed receptor image is static. The
skin format also has no way to show lighting only for GREAT and PERFECT hits.

The splash art stays relatively large in the PNG for detail. `Skin.ini` sets a
small `LightingNWidth` to draw it at note size, and the converter renders the
`@2x` frames directly from the atlas to avoid enlarging the SD frames.
Hold tails include a light dot at the release point. The stretched hold body
provides the background behind the dot.
