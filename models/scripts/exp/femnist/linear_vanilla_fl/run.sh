date

num_rounds=3000
batch_size=10
num_epochs=1
clients_per_round=100
lr=1.0
reg=0.000000001
model="erm_log_reg"

dataset="femnist"

outf="outputs/exp/femnist/linear_vanilla_fl/"
logf="outputs/exp/femnist/linear_vanilla_fl/logs"

main_args=" -dataset ${dataset} -model ${model}"
options=" --num-rounds ${num_rounds} -lr ${lr} --eval-every 50 --clients-per-round ${clients_per_round} --num_epochs ${num_epochs} -reg_param $reg --full_record True"

seed=1
echo "starting seed : $seed"
time python main.py ${main_args} ${options} --seed ${seed}  --output_summary_file ${outf}seed_${seed}  > ${logf}_seed_${seed} 2>&1

seed=2
echo "starting seed : $seed"
time python main.py ${main_args} ${options} --seed ${seed}  --output_summary_file ${outf}seed_${seed}  > ${logf}_seed_${seed} 2>&1

seed=3
echo "starting seed : $seed"
time python main.py ${main_args} ${options} --seed ${seed}  --output_summary_file ${outf}seed_${seed}  > ${logf}_seed_${seed} 2>&1

seed=4
echo "starting seed : $seed"
time python main.py ${main_args} ${options} --seed ${seed}  --output_summary_file ${outf}seed_${seed}  > ${logf}_seed_${seed} 2>&1

seed=5
echo "starting seed : $seed"
time python main.py ${main_args} ${options} --seed ${seed}  --output_summary_file ${outf}seed_${seed}  > ${logf}_seed_${seed} 2>&1
