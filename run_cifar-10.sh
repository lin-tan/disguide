cd disguide;

read -p "Device to use:" device
read -p "Select hard or soft label, one of [hl,l1]" loss
echo "Running DisGUIDE on CIFAR10"

dataset=cifar10              # Dataset to run on
#loss=hl                      # hl for hard-label, l1 for soft-label
teacher_arch=resnet34        # Supports resnet34 and resnet18
replay="Classic"             # Standard experience replay. Set to "Off" to disable replay
input_space="pre-transform"  # "pre-transform" implies no attacker knowledge. "post-transform" assumes knowledge of image preprocessing
query_budget=20              # Query budget in millions
lr_S=0.03                    # Initial student learning rate
ld=2                         # Set to 2 for cifar10, 04 for cifar100
ri=3                         # Replay training iterations
rs=100                       # Replay size in 10k
grayscale=8                  # Set to 0 to deactivate grayscale, 8 to enable
experiment_type="disguide"   # Whether to run disguide or dfme experiment
ensemble_size=2              # Value must be 2 or higher for DisGUIDE. May be any value for DFME
suffix="${dataset}_SlrS${lrDec}qb${query_budget}st048di1ld0${ld}gs${grayscale}ri${ri}rs${rs}"

python3 train.py --experiment-type ${experiment_type} --epoch-itr 150 --log-interval 30 --d-iter 1 --grayscale ${grayscale} --rep-iter ${ri} --replay-size ${rs}0000  --input-space ${input_space} --lambda-div 0.${ld} --model ${teacher_arch}_8x --step 0.4 0.8 --ckpt checkpoint/teacher/${dataset}-${teacher_arch}_8x.pt --dataset ${dataset} --lr-S ${lr_S} --suffix ${suffix} --replay ${replay} --device ${device} --query-budget ${query_budget} --log-dir save_results/${dataset}  --lr-G 1e-4 --ensemble-size ${ensemble_size} --batch-size 256 --loss ${loss} --student-model ensemble_resnet18_8x;
