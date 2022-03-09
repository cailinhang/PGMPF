
#python ./utils/get_small_model.py /home/user/winycg/dataset/ImageNet  -a resnet50  --workers 12 \
#--resume  /home/user/clh/SFP/logs/resnet50-rate-0.6/best.resnet50.2019-12-26-8658.pth.tar \
#--save_dir ./small/resnet50/ --batch-size 128 \
#--get_small

#python ./utils/get_small_model.py /home/user/winycg/dataset/ImageNet  -a resnet50  --workers 12 \
#--resume  /home/user/clh/SFP/logs/resnet50-rate-0.7/checkpoint.resnet50.2020-01-12-200.pth.tar \
#--save_dir ./small/resnet50/ --batch-size 128 \
#--get_small

#python ./utils/get_small_model.py /home/user/winycg/dataset/ImageNet  -a resnet50  --workers 12 \
#--resume  /home/user/clh/SFP/logs/resnet50-rate-0.7/checkpoint.resnet50.2020-07-08-9661.pth.tar \
#--save_dir ./small/resnet50/ --batch-size 128 \
#--get_small

#python ./utils/get_small_model.py /home/user/winycg/dataset/ImageNet  -a resnet50  --workers 12 \
#--resume  /home/user/clh/momentum_pruning/logs/resnet50-rate-0.6/checkpoint.resnet50.2021-09-25-4028.pth.tar \
#--save_dir ./small/resnet50/ --batch-size 128 \
#--get_small


#python ./utils/get_small_model.py /home/user/winycg/dataset/ImageNet  -a resnet50  --workers 12 \
#--resume  /home/user/clh/momentum_pruning/logs/resnet50-rate-0.6/checkpoint.resnet50.2021-06-21-1998_epoch_89_backup.pth.tar \
#--save_dir ./small/resnet50/ --batch-size 128 \
#--get_small


#python ./utils/get_small_model.py /home/user/winycg/dataset/ImageNet  -a resnet50  --workers 12 \
#--resume  /home/user/clh/momentum_pruning/logs/resnet50-rate-0.6/checkpoint.resnet50.2021-12-20-9349.pth.tar \
#--save_dir ./small/resnet50/ --batch-size 128 \
#--get_small

# trained 150 epoches with PGMPF, then prune the bn, and fine-tune one epoch, 
# successfully converted to pruned small model
#python ./utils/get_small_model.py /home/user/winycg/dataset/ImageNet  -a resnet50  --workers 12 \
#--resume  /home/user/clh/momentum_pruning/logs/resnet50-rate-0.6/checkpoint.resnet50.2021-12-22-7259.pth.tar \
#--save_dir ./small/resnet50/ --batch-size 128 \
#--get_small

#python ./utils/get_small_model.py /home/user/winycg/dataset/ImageNet  -a resnet34  --workers 12 \
#--resume  /home/user/clh/momentum_pruning/logs/resnet34-rate-0.6/checkpoint.resnet34.2021-07-10-4320.pth.tar \
#--save_dir ./small/resnet34/ --batch-size 128 \
#--get_small
# sfp
#ckpt=checkpoint.resnet34.2020-01-04-8678.pth.tar
#ckpt=checkpoint.resnet34.2019-12-20-2656.pth.tar
# pgmpf
#ckpt=checkpoint.resnet34.2021-07-10-9947.pth.tar
# mask bn
#ckpt=checkpoint.resnet34.2021-12-23-5756.pth.tar
# train with pure mask_bn
ckpt=checkpoint.resnet34.2021-12-25-6993.pth.tar 

save_path=/home/user/clh/SFP/logs/resnet34-rate-0.6/
save_path=/home/user/clh/momentum_pruning/logs/resnet34-rate-0.6/

python ./utils/get_small_model.py /home/user/winycg/dataset/ImageNet  -a resnet34  --workers 12 \
--resume  ${save_path}/${ckpt} \
--save_dir ./small/resnet34/ --batch-size 128 \
--get_small
