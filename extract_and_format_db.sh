#!/bin/bash
DATASET_ZIP=${1:-"./data/"}
ROOT_FOLDER=$(dirname $DATASET_ZIP)

set -e

PREV_DIR=$(pwd)

cd $ROOT_FOLDER
unzip IDMT-SMT-AUDIO-EFFECTS.zip

mv IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/* IDMT-SMT-AUDIO-EFFECTS/
rm -r IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/
cd IDMT-SMT-AUDIO-EFFECTS
rm Gitarre_polyphon2.zip

mv 'Bass monophon2.zip' Bass_monophon2.zip
mv 'Bass monophon.zip' Bass_monophon.zip
mv 'Gitarre monophon2.zip' Gitarre_monophon2.zip
mv 'Gitarre monophon.zip' Gitarre_monophon.zip
mv 'Gitarre polyphon2.zip' Gitarre_polyphon2.zip
mv 'Gitarre polyphon.zip' Gitarre_polyphon.zip

for file in $(ls *.zip); do unzip $file; done
rm *.zip

mv Bass\ monophon Bass_monophon
mv Bass\ monophon2 Bass_monophon2
mv Gitarre\ monophon Gitarre_monophon
mv Gitarre\ monophon2 Gitarre_monophon2
mv Gitarre\ polyphon Gitarre_polyphon
mv Gitarre\ polyphon2 Gitarre_polyphon2

cp -r Bass_monophon2/Samples/* Bass_monophon/Samples
cp -r Gitarre_monophon2/Samples/* Gitarre_monophon/Samples
cp -r Gitarre_polyphon2/* Gitarre_polyphon/Samples

rm -rf Gitarre_polyphon2 Gitarre_monophon2 Bass_monophon2

cd $PREV_DIR
