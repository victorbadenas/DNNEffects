experiment=$args[0]
if ($null -eq ${experiment}) {
    $experiment="test"
}
New-Item -Name experiments/ -ItemType directory
New-Item -Name experiments/${experiment} -ItemType directory
logFile=../experiments/${experiment}/train.log
Set-Location ./src/
python train.py -n ${experiment} --log_file ${logFile}
Set-Location ..
