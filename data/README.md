# External MAESTRO Data

Do not place the original dataset in the training package archive. Point the
training commands at an extracted MAESTRO v3.0.0 directory on the training
server instead. The directory must contain the metadata CSV and every MIDI/WAV
path named by that CSV:

```text
maestro-v3.0.0/
  maestro-v3.0.0.csv
  2004/
    ... .midi
    ... .wav
  ...
```

The MIDI-only MAESTRO copy in the parent project is not sufficient for DDSP
training. The target waveform is supervised by the aligned WAV files.
