# Kill current process and restart with reduced beam
python scripts/eval_offline.py \
  --modelPath data/models/rnn_model \
  --datasetPath data/pickledData/ptDecoder_ctc.pkl \
  --lmPath data/models/three_gram_lm \
  --device cuda \
  --beam 8  # Reduced from 18