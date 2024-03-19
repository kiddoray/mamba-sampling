CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python $(dirname "$0")/train.py $CONFIG --launcher none
