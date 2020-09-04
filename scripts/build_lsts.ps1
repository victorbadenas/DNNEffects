$array = "Chorus","Distortion","EQ","FeedbackDelay","Flanger","Overdrive","Phaser","Reverb","SlapbackDelay","Tremolo","Vibrato"
$array | ForEach-Object {} {
    python createLstDataset.py -i E:/IDMT-SMT-AUDIO-EFFECTS/ -e $_
} {}
