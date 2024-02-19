mkdir -p logs
datasets="all"
for num_token in {1,10}; do  
  for n_frame_per_clip in 16; do
    for join_length in {-1,100,300,600}; do
        max_num_frm=$((n_frame_per_clip * num_token))
        output_json="logs/mvbench_lvchat_${n_frame_per_clip}x${num_token}.json"
        output_txt="logs/mvbench_lvchat_${n_frame_per_clip}x${num_token}.txt"
        python mvbench.py $@ --max_num_frm ${max_num_frm} --datasets "${datasets}" --n_frame_per_clip ${n_frame_per_clip} --output_file ${output_json} > ${output_txt}
    done
  done
done