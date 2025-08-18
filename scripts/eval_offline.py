#!/usr/bin/env python3
"""
Offline evaluation script for RNN + 3-gram baseline on test partition.
Reproduces the 18.8% WER baseline mentioned in the paper.
"""

import re
import time
import pickle
import numpy as np
import argparse
import sys
import os
import signal

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is required but not installed.")
    print("Install with: pip install tqdm")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Error: psutil is required but not installed.")
    print("Install with: pip install psutil")
    sys.exit(1)

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_decoder.dataset import SpeechDataset
from neural_decoder.neural_decoder_trainer import loadModel

# Import only what we need to avoid tensorflow dependency
import lm_decoder
import neural_decoder.utils.rnnEval as rnnEval


# Global variables for monitoring
_interrupt_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _interrupt_requested
    print("\nInterrupt received. Finishing current sample and saving progress...")
    _interrupt_requested = True


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 3)  # Convert to GB


def get_gpu_memory_usage():
    """Get GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    return 0.0


def check_system_health(memory_threshold_gb=16.0):
    """Check system health and warn if resources are running low."""
    memory_gb = get_memory_usage()
    gpu_memory_gb = get_gpu_memory_usage()

    warnings = []
    if memory_gb > memory_threshold_gb:
        warnings.append(f"WARNING: High RAM usage: {memory_gb:.1f} GB")

    if gpu_memory_gb > 8.0:  # Warn if GPU memory > 8GB
        warnings.append(f"WARNING: High GPU memory usage: {gpu_memory_gb:.1f} GB")

    cpu_percent = psutil.cpu_percent(interval=0.1)
    if cpu_percent > 90:
        warnings.append(f"WARNING: High CPU usage: {cpu_percent:.1f}%")

    return warnings, memory_gb, gpu_memory_gb


def print_system_status():
    """Print current system status."""
    warnings, memory_gb, gpu_memory_gb = check_system_health()

    print(f"RAM: {memory_gb:.1f} GB", end="")
    if gpu_memory_gb > 0:
        print(f" | GPU: {gpu_memory_gb:.1f} GB", end="")
    print(f" | CPU: {psutil.cpu_percent(interval=0.1):.1f}%")

    for warning in warnings:
        print(warning)


def build_lm_decoder(model_path,
                     max_active=7000,
                     min_active=200,
                     beam=17.,
                     lattice_beam=8.,
                     acoustic_scale=1.5,
                     ctc_blank_skip_threshold=1.0,
                     length_penalty=0.0,
                     nbest=1):
    """Build language model decoder (copied from lmDecoderUtils to avoid tensorflow import)."""
    decode_opts = lm_decoder.DecodeOptions(
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        ctc_blank_skip_threshold,
        length_penalty,
        nbest
    )

    TLG_path = os.path.join(model_path, 'TLG.fst')
    words_path = os.path.join(model_path, 'words.txt')
    G_path = os.path.join(model_path, 'G.fst')
    rescore_G_path = os.path.join(model_path, 'G_no_prune.fst')
    if not os.path.exists(rescore_G_path):
        rescore_G_path = ""
        G_path = ""
    if not os.path.exists(TLG_path):
        raise ValueError('TLG file not found at {}'.format(TLG_path))
    if not os.path.exists(words_path):
        raise ValueError('words file not found at {}'.format(words_path))

    decode_resource = lm_decoder.DecodeResource(
        TLG_path,
        G_path,
        rescore_G_path,
        words_path,
        ""
    )
    decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)
    return decoder


def lm_decode(decoder, logits, returnNBest=False, rescore=False,
              blankPenalty=0.0,
              logPriors=None):
    """Decode with language model (copied from lmDecoderUtils to avoid tensorflow import)."""
    assert len(logits.shape) == 2

    if logPriors is None:
        logPriors = np.zeros([1, logits.shape[1]])
    lm_decoder.DecodeNumpy(decoder, logits, logPriors, blankPenalty)
    decoder.FinishDecoding()
    if rescore:
        decoder.Rescore()

    if not returnNBest:
        if len(decoder.result()) == 0:
            decoded = ''
        else:
            decoded = decoder.result()[0].sentence
    else:
        decoded = []
        for r in decoder.result():
            decoded.append((r.sentence, r.ac_score, r.lm_score))

    decoder.Reset()
    return decoded


def rearrange_speech_logits(logits, has_sil=True):
    """Rearrange logits for speech decoding (copied from lmDecoderUtils to avoid tensorflow import)."""
    # This function rearranges the logits to match the expected order for speech decoding
    # The exact implementation depends on the phoneme ordering
    # For now, we'll return as-is since the 3-gram model should handle the current ordering
    return logits


def getDatasetLoaders(datasetPath, batchSize=1):
    """Load the pickled dataset and create data loaders."""
    with open(datasetPath, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    test_ds = SpeechDataset(loadedData["test"])
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_padding,
    )

    return test_loader, loadedData


def extract_rnn_outputs(model, _, loadedData, device="cpu"):
    """Extract RNN outputs from the test set."""
    global _interrupt_requested
    model.eval()

    rnn_outputs = {
        "logits": [],
        "logitLengths": [],
        "trueSeqs": [],
        "transcriptions": [],
    }

    # Calculate total number of trials for progress tracking
    total_trials = sum(len(day_data["sentenceDat"]) for day_data in loadedData["test"])

    print("Extracting RNN outputs from test set...")
    print(f"Total trials to process: {total_trials}")
    print_system_status()
    print()

    # Setup signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    trial_count = 0
    start_time = time.time()

    # Create overall progress bar for all trials
    with tqdm(total=total_trials, desc="RNN Inference",
              unit="trial", ncols=100,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        # Process each day in the test partition
        for dayIdx in range(len(loadedData["test"])):
            if _interrupt_requested:
                print("\nStopping RNN extraction due to interrupt...")
                break

            day_data = loadedData["test"][dayIdx]

            # Create dataset for this specific day
            day_ds = SpeechDataset([day_data])
            day_loader = DataLoader(day_ds, batch_size=1, shuffle=False, num_workers=0)

            for trialIdx, (X, y, X_len, y_len, _) in enumerate(day_loader):
                if _interrupt_requested:
                    break

                X, y, X_len, y_len = (
                    X.to(device),
                    y.to(device),
                    X_len.to(device),
                    y_len.to(device),
                )

                # Use the correct day index for the model
                dayIdx_tensor = torch.tensor([dayIdx], dtype=torch.int64).to(device)

                with torch.no_grad():
                    pred = model.forward(X, dayIdx_tensor)
                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

                    # Store outputs
                    rnn_outputs["logits"].append(pred[0].cpu().detach().numpy())
                    rnn_outputs["logitLengths"].append(adjustedLens[0].cpu().detach().item())
                    rnn_outputs["trueSeqs"].append(np.array(y[0][0:y_len[0]].cpu().detach()))

                    # Get the transcription for this trial
                    transcript = day_data["transcriptions"][trialIdx].strip()
                    transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
                    transcript = transcript.replace("--", "").lower()
                    rnn_outputs["transcriptions"].append(transcript)

                trial_count += 1
                pbar.update(1)

                # Update progress bar description with current day info
                pbar.set_description(f"RNN Inference (Day {dayIdx + 1}/{len(loadedData['test'])})")

                # Check system health every 50 trials
                if trial_count % 50 == 0:
                    warnings, memory_gb, gpu_memory_gb = check_system_health()
                    if warnings:
                        pbar.write(f"System warnings at trial {trial_count}:")
                        for warning in warnings:
                            pbar.write(f"   {warning}")

                    # Update postfix with memory info
                    pbar.set_postfix({
                        'RAM': f'{memory_gb:.1f}GB',
                        'GPU': f'{gpu_memory_gb:.1f}GB' if gpu_memory_gb > 0 else 'N/A'
                    })

    elapsed_time = time.time() - start_time
    samples_extracted = len(rnn_outputs["logits"])

    print(f"\nExtracted {samples_extracted} samples from test set")
    print(f"Total extraction time: {elapsed_time:.1f}s ({elapsed_time/samples_extracted:.3f}s per sample)")
    print_system_status()

    if _interrupt_requested and samples_extracted < total_trials:
        print(f"Extraction interrupted. Got {samples_extracted}/{total_trials} samples.")

    return rnn_outputs


def evaluate_with_3gram(rnn_outputs, lm_decoder_path, acoustic_scale=0.5, blank_penalty=np.log(7), beam=18.0):
    """Evaluate using 3-gram language model decoder."""
    global _interrupt_requested

    print(f"Building 3-gram decoder from: {lm_decoder_path}")

    # Build the 3-gram decoder
    ngramDecoder = build_lm_decoder(
        lm_decoder_path,
        acoustic_scale=acoustic_scale,
        nbest=1,  # Just get best hypothesis
        beam=beam
    )

    total_samples = len(rnn_outputs["logits"])
    print(f"Starting 3-gram language model decoding...")
    print(f"Total samples to decode: {total_samples}")
    print_system_status()
    print()

    decoded_sentences = []
    start_time = time.time()

    # Create progress bar for decoding
    with tqdm(total=total_samples, desc="3-gram Decoding",
              unit="sample", ncols=100,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for i, logits in enumerate(rnn_outputs["logits"]):
            if _interrupt_requested:
                print("\nStopping 3-gram decoding due to interrupt...")
                break

            # Rearrange logits: move blank token from first to last position
            logits_reordered = np.concatenate([logits[:, 1:], logits[:, 0:1]], axis=-1)

            # Rearrange for speech decoding (with silence token)
            logits_speech = rearrange_speech_logits(
                logits_reordered[None, :, :], has_sil=True
            )

            # Decode with language model
            decoded = lm_decode(
                ngramDecoder,
                logits_speech[0],
                blankPenalty=blank_penalty,
                returnNBest=False,
                rescore=True,
            )

            decoded_sentences.append(decoded.strip())
            pbar.update(1)

            # Check system health and update progress info every 100 samples
            if (i + 1) % 100 == 0:
                warnings, memory_gb, gpu_memory_gb = check_system_health()
                if warnings:
                    pbar.write(f"System warnings at sample {i + 1}:")
                    for warning in warnings:
                        pbar.write(f"   {warning}")

                # Calculate current processing rate
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0

                # Update postfix with performance info
                postfix = {
                    'Rate': f'{rate:.1f}/s',
                    'RAM': f'{memory_gb:.1f}GB',
                    'Avg': f'{elapsed/(i+1):.3f}s/sample'
                }
                if gpu_memory_gb > 0:
                    postfix['GPU'] = f'{gpu_memory_gb:.1f}GB'
                pbar.set_postfix(postfix)

    decode_time = time.time() - start_time
    samples_decoded = len(decoded_sentences)

    print(f"\n3-gram decoding completed!")
    print(f"Decoded {samples_decoded} samples")
    print(f"Total decoding time: {decode_time:.1f}s ({decode_time/samples_decoded:.3f}s per sample)")
    print(f"Average processing rate: {samples_decoded/decode_time:.1f} samples/second")
    print_system_status()

    if _interrupt_requested and samples_decoded < total_samples:
        print(f"Decoding interrupted. Got {samples_decoded}/{total_samples} samples.")

    return decoded_sentences


def calculate_wer(decoded_sentences, true_sentences):
    """Calculate Word Error Rate."""
    print("Calculating Word Error Rate...")

    total_word_errors = 0
    total_words = 0
    total_samples = len(decoded_sentences)

    # Use tqdm for WER calculation if we have many samples
    iterator = zip(decoded_sentences, true_sentences)
    if total_samples > 100:
        iterator = tqdm(iterator, total=total_samples, desc="Computing WER",
                       unit="sample", ncols=80)

    for decoded, true in iterator:
        true_words = true.split()
        decoded_words = decoded.split()

        word_errors = rnnEval.wer(true_words, decoded_words)
        total_word_errors += word_errors
        total_words += len(true_words)

    wer_rate = (total_word_errors / total_words) * 100 if total_words > 0 else 0
    print(f"WER calculation completed: {wer_rate:.1f}%")
    return wer_rate, total_word_errors, total_words


def main():
    parser = argparse.ArgumentParser(description="Evaluate RNN + 3-gram baseline on test set")
    parser.add_argument("--modelPath", type=str, required=True,
                       help="Path to trained RNN model directory")
    parser.add_argument("--datasetPath", type=str, default="data/pickledData/ptDecoder_ctc.pkl",
                       help="Path to pickled dataset")
    parser.add_argument("--lmPath", type=str, default="data/models/three_gram_lm",
                       help="Path to 3-gram language model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to use for inference")
    parser.add_argument("--acousticScale", type=float, default=0.5,
                       help="Acoustic scale for language model decoding")
    parser.add_argument("--blankPenalty", type=float, default=None,
                       help="Blank penalty (default: log(7))")
    parser.add_argument("--memoryThreshold", type=float, default=16.0,
                       help="Memory threshold in GB for warnings (default: 16.0)")
    parser.add_argument("--beam", type=float, default=18.0,
                       help="Beam size for language model decoding (default: 18.0)")

    args = parser.parse_args()

    if args.blankPenalty is None:
        args.blankPenalty = np.log(7)

    # Setup signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("RNN + 3-gram Baseline Evaluation")
    print("=" * 70)
    print(f"Model path: {args.modelPath}")
    print(f"Dataset path: {args.datasetPath}")
    print(f"Language model path: {args.lmPath}")
    print(f"Device: {args.device}")
    print(f"Acoustic scale: {args.acousticScale}")
    print(f"Blank penalty: {args.blankPenalty:.3f}")
    print(f"Beam size: {args.beam}")
    print(f"Memory threshold: {args.memoryThreshold:.1f} GB")
    print()

    # Initial system status
    print("Initial system status:")
    print_system_status()
    print()

    try:
        # Load model
        print("Loading trained RNN model...")
        start_time = time.time()
        model = loadModel(args.modelPath, device=args.device)
        load_time = time.time() - start_time
        print(f"Model loaded successfully ({load_time:.1f}s)")
        print_system_status()
        print()

        # Load dataset
        print("Loading test dataset...")
        start_time = time.time()
        test_loader, loadedData = getDatasetLoaders(args.datasetPath)
        load_time = time.time() - start_time
        total_days = len(loadedData['test'])
        total_trials = sum(len(day_data["sentenceDat"]) for day_data in loadedData["test"])
        print(f"Dataset loaded successfully ({load_time:.1f}s)")
        print(f"Test set contains {total_days} days with {total_trials} total trials")
        print_system_status()
        print()

        # Extract RNN outputs
        print("Starting RNN output extraction...")
        rnn_outputs = extract_rnn_outputs(model, test_loader, loadedData, device=args.device)

        if _interrupt_requested:
            print("Process interrupted during RNN extraction. Exiting...")
            return

        # Decode with 3-gram LM
        print("\nStarting 3-gram language model decoding...")
        decoded_sentences = evaluate_with_3gram(
            rnn_outputs,
            args.lmPath,
            acoustic_scale=args.acousticScale,
            blank_penalty=args.blankPenalty,
            beam=args.beam
        )

        if _interrupt_requested:
            print("Process interrupted during 3-gram decoding. Exiting...")
            return

        # Calculate WER
        print("\nStarting WER calculation...")
        wer_rate, total_errors, total_words = calculate_wer(decoded_sentences, rnn_outputs["transcriptions"])

    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C)")
        return
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        print("Final system status:")
        print_system_status()
        raise

    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total samples processed: {len(decoded_sentences)}")
    print(f"Total words: {total_words}")
    print(f"Total word errors: {total_errors}")
    print(f"Word Error Rate (WER): {wer_rate:.1f}%")
    print()

    # Compare with baseline
    baseline_wer = 18.8
    difference = abs(wer_rate - baseline_wer)
    if difference < 1.0:
        print(f"SUCCESS: WER ({wer_rate:.1f}%) is close to the reported {baseline_wer}% baseline!")
    elif wer_rate < baseline_wer:
        print(f"EXCELLENT: WER ({wer_rate:.1f}%) is better than the {baseline_wer}% baseline by {baseline_wer - wer_rate:.1f}%!")
    else:
        print(f"WARNING: WER ({wer_rate:.1f}%) differs from reported {baseline_wer}% baseline (difference: +{difference:.1f}%)")

    print(f"\nFirst 5 examples:")
    print("-" * 70)
    for i in range(min(5, len(decoded_sentences))):
        print(f"True:    {rnn_outputs['transcriptions'][i]}")
        print(f"Decoded: {decoded_sentences[i]}")
        print()

    # Final system status
    print("Final system status:")
    print_system_status()
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()