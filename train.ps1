$experiment=$args[0]
if ($null -eq ${experiment}) {
    $experiment="test"
}
$effect="Distortion"
$trainLst="./dataset/${effect}/train.lst"
$testLst="./dataset/${effect}/test.lst"
New-Item -Name experiments/ -ItemType directory
New-Item -Name experiments/${experiment} -ItemType directory
$logFile="../experiments/${experiment}/train.log"
python src/train.py -n ${experiment} --log_file ${logFile} -trainlst ${trainLst} -testlst ${testLst}
