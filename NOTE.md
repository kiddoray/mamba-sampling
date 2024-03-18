## train
nohup bash utils/single_gpu_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py > train_cls_local_origin_log 2>&1 &

## test
nohup bash utils/single_gpu_test.sh configs/apes/apes_cls_local-modelnet-200epochs.py work_dirs/apes_cls_local-modelnet-200epochs/20240315_191504/best_val_acc_epoch_184.pth > test_cls_local_0316_log 2>&1 &

work_dirs/apes_cls_local-modelnet-200epochs/20240317_015926/best_val_acc_epoch_168.pth

work_dirs/apes_cls_local-modelnet-200epochs/20240317_140546/best_val_acc_epoch_140.pth