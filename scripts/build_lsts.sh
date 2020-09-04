for $effect in $(echo Chorus Distortion EQ FeedbackDelay Flanger Overdrive Phaser Reverb SlapbackDelay Tremolo Vibrato); do
    python createLstDataset.py -i E:/IDMT-SMT-AUDIO-EFFECTS/ -e $effect
done;
