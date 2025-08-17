#!/bin/bash

# Full RNN training
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PICKLE_PATH="${PROJECT_ROOT}/data/pickledData/ptDecoder_ctc.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/models/rnn_model"
LOG_DIR="${PROJECT_ROOT}/logs/rnn_model"

echo "Full training - RNN model"
echo "Using data: $PICKLE_PATH"
echo "Model output: $OUTPUT_DIR"
echo "Logs output: $LOG_DIR"
echo ""
echo "WARNING: Training will create files in visible locations:"
echo "  - Model files: $OUTPUT_DIR"
echo "  - Training logs: $LOG_DIR"
echo "  - This will take several hours and create large files!"
echo "  - Monitor disk space during training!"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

cat > /tmp/train_full.py << EOF
import sys
sys.path.append('${PROJECT_ROOT}/src')

args = {}
args['outputDir'] = '${OUTPUT_DIR}'
args['datasetPath'] = '${PICKLE_PATH}'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel
trainModel(args)
EOF

cd "$PROJECT_ROOT"
python /tmp/train_full.py 2>&1 | tee "$LOG_DIR/training.log"
rm /tmp/train_full.py

echo ""
echo "Full training completed!"
echo "Files created:"
echo "  Model: $OUTPUT_DIR"
echo "  Logs: $LOG_DIR/training.log"
echo ""
echo "To check disk usage: du -sh $OUTPUT_DIR $LOG_DIR"
