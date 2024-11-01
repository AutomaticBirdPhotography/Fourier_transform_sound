# Sonett18

Se complete/sonett18.wav

```
$ python text_to_sound.py
Enter the text to be converted to signal (default: hello):
Enter the duration of each character in seconds (defualt: 0.1): 0.01
Enter the base frequency (default: 1000): 100
Enter the bitrate (default: 44100):
Enter the threshold (default: 0.5):
Enter the filename (default: signal.wav): sonet18.wav
Signal generated and saved in the file: 'sonet18.wav'
```

```
$ python sound_to_text.py
Enter the duration of each character in seconds (defualt: 0.1): 0.01
Enter the base frequency (default: 1000): 100
Enter the threshold (default: 0.5):
Enter the filename (default: ./signal.wav): ./sonet18.wav
Decoded message:
        Sonett 18
        Skal jeg si du er lik en sommerdag?
        Du har mer ynde og mer harmoni.
        Hver maiknopp skakes bryskt av vindens jag,
        og sommerens lånte tid er fort forbi.

        For het iblant er himmeløyets glød,
        og ofte blir dets gylne glans obskur.
        Alt skjønt går fra det skjønne og mot død,
        på grunn av skjebne eller streng natur.

        Men alltid skal din sommer finnes her,
        din skjønnhet skal bestå, du skal forbli,
        og ikke gå hvor dødens skygge er:

        Av diktet, evig, oppstår du i tid.
        ___Så lenge menn kan ånde, øyne se,
        ___har diktet liv. Og du får liv ved det.
```
