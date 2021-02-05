name=${1:-"test"}
effect=${2:-"Distortion"}
batch_size=${3:-"64"}
frame_length=${4:-"4096"}
epochs=${5:-"300"}

logFile=models/${name}/logs/train.log
python src/train.py -n ${name} --train_lst dataset/${effect}/train.lst --test_lst dataset/${effect}/test.lst --log_file ${logFile} --batch_size ${batch_size} --frame_length ${frame_length} --epochs ${epochs}
