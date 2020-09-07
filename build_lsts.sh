for effect in $(echo Chorus Distortion EQ FeedbackDelay Flanger Overdrive Phaser Reverb SlapbackDelay Tremolo Vibrato);
do
    python scripts/createLstDataset.py -i ./data/ -o ./dataset/ --log_file ./log/$effect.log -e ${effect};
    python scripts/splitLstTrainTest.py -i ./dataset/${effect}.lst -test 0.1
done;
