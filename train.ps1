$experiment=$args[0]
if ($null -eq ${experiment}) {
    $experiment="test"
}
$effect="Distortion"
$trainLst="./dataset/${effect}/train.lst"
$testLst="./dataset/${effect}/test.lst"
$batchSize=256
$frameLength=128
$logFile="./experiments/${experiment}/train.log"
python src/train.py -n ${experiment} --log_file ${logFile} -trainlst ${trainLst} -testlst ${testLst} --batch_size=$batchSize --frame_length=$frameLength
