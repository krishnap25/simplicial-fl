date

gpu=1

model="erm_cnn_log_reg"
dataset="femnist"

num_rounds=1000
batch_size=16
num_epochs=1
clients_per_round=100
eval_every=10
lr=0.32
lr_decay=2
decay_lr_every=400
run_simplicial_fl=True
nonconformity_level=0.9

outf="outputs/exp/femnist/nn_simplicial_fl/${nonconformity_level}_"
logf="outputs/exp/femnist/nn_simplicial_fl/logs_${nonconformity_level}_"
full_record=True

main_args=" -dataset ${dataset} -model ${model}"
options=" --num-rounds ${num_rounds} \
          -lr ${lr} --lr-decay ${lr_decay} --decay-lr-every ${decay_lr_every} \
          --eval-every ${eval_every} \
          --clients-per-round ${clients_per_round} \
          --num_epochs ${num_epochs} \
          --full_record ${full_record} \
          --batch_size ${batch_size} \
          --run_simplicial_fl ${run_simplicial_fl} \
          --nonconformity_level ${nonconformity_level} \
          --gpu ${gpu}"

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
