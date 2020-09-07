$array = "Chorus","Distortion","EQ","FeedbackDelay","Flanger","Overdrive","Phaser","Reverb","SlapbackDelay","Tremolo","Vibrato"
$array | ForEach-Object {} {
    python scripts/createLstDataset.py -i E:/IDMT-SMT-AUDIO-EFFECTS/ -e $_;
    python scripts/splitLstTrainTest.py -i ./dataset/$_.lst -test 0.1
} {}
