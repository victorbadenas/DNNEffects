experiment=${1:-"test"}
mkdir experiments/
mkdir experiments/${experiment}
logFile=../experiments/${experiment}/train.log
cd ./src/
python train.py -n ${experiment} --log_file ${logFile}
cd ..
