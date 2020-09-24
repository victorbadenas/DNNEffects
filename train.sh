experiment=${1:-"test"}
logFile=../experiments/${experiment}/train.log
cd ./src/
python train.py -n ${experiment} --log_file ${logFile}
cd ..
