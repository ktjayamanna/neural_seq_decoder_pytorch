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
echo "  - TensorBoard logs: $OUTPUT_DIR/tensorboard_logs"
echo "  - This will take several hours and create large files!"
echo "  - Monitor disk space during training!"
echo ""
echo "MONITORING TRAINING PROGRESS:"
echo "  To view training metrics with TensorBoard:"
echo "  1. Open a new terminal"
echo "  2. Run: tensorboard --logdir $OUTPUT_DIR/tensorboard_logs"
echo "  3. Open http://localhost:6006 in your browser"
echo "  4. You'll see training loss and validation CER"
echo ""
echo "NOTE: TensorBoard logs are minimal and overwrite previous runs to save space"
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
args['batchSize'] = 32
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
echo "  TensorBoard logs: $OUTPUT_DIR/tensorboard_logs"
echo ""
echo "To check disk usage: du -sh $OUTPUT_DIR $LOG_DIR"
echo ""
echo "To view training metrics:"
echo "  tensorboard --logdir $OUTPUT_DIR/tensorboard_logs"
echo "  Then open http://localhost:6006 in your browser"
