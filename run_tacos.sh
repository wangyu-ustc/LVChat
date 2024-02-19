mkdir -p logs

for num_token in {1,6}; do  
  for n_frame_per_clip in 16; do
        max_num_frm=$((n_frame_per_clip * num_token))
        output_json="logs/tacos_lvchat_${n_frame_per_clip}x${num_token}.json"
        output_txt="logs/tacos_lvchat_${n_frame_per_clip}x${num_token}.txt"
        python inference.py $@ --max_num_frm ${max_num_frm}  --n_frame_per_clip ${n_frame_per_clip} --output_file ${output_json} > ${output_txt}
  done
done

