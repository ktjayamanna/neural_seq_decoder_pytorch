#!/bin/bash

# Quick RNN training for testing
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PICKLE_PATH="${PROJECT_ROOT}/data/pickledData/ptDecoder_ctc.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/models/rnn_model_quick"
LOG_DIR="${PROJECT_ROOT}/logs/rnn_model_quick"

echo "Quick training - RNN model"
echo "Using data: $PICKLE_PATH"
echo "Model output: $OUTPUT_DIR"
echo "Logs output: $LOG_DIR"
echo ""
echo "WARNING: Training will create files in visible locations:"
echo "  - Model files: $OUTPUT_DIR"
echo "  - Training logs: $LOG_DIR"
echo "  - Monitor disk space during training!"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

cat > /tmp/train_quick.py << EOF
import sys
sys.path.append('${PROJECT_ROOT}/src')

args = {}
args['outputDir'] = '${OUTPUT_DIR}'
args['datasetPath'] = '${PICKLE_PATH}'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 32
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 512
args['nBatch'] = 500
args['nLayers'] = 3
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
python /tmp/train_quick.py 2>&1 | tee "$LOG_DIR/training.log"
rm /tmp/train_quick.py

echo ""
echo "Quick training completed!"
echo "Files created:"
echo "  Model: $OUTPUT_DIR"
echo "  Logs: $LOG_DIR/training.log"
echo ""
echo "To check disk usage: du -sh $OUTPUT_DIR $LOG_DIR"
