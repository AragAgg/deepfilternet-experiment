#!/usr/bin/env python3
"""
DeepFilterNet Speech Enhancement Experiment.
Tests DeepFilterNet on audio files and measures RTF (Real-Time Factor).

DeepFilterNet is a low-complexity speech enhancement model that removes
background noise while preserving speech quality.

Usage:
    python deepfilternet-experiment/test_deepfilternet.py
    python deepfilternet-experiment/test_deepfilternet.py --input-dir ./some_audio/
    python deepfilternet-experiment/test_deepfilternet.py single_file.wav

Environment variables:
    DF_MODEL: Model to use (default: DeepFilterNet3)
"""

import os
import sys
import time
import glob
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import librosa
import soundfile as sf

# DeepFilterNet imports
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    from df import config
except ImportError:
    print("Error: DeepFilterNet not installed. Run: pip install deepfilternet")
    sys.exit(1)

# Directory setup
SCRIPT_DIR = Path(__file__).parent
TEST_AUDIO_DIR = SCRIPT_DIR / "test_audio"
OUTPUT_DIR = SCRIPT_DIR / "output"


@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    input_file: str
    output_file: str
    duration_sec: float
    processing_time_sec: float
    rtf: float  # Real-Time Factor (< 1 means faster than real-time)
    input_sr: int
    
    @property
    def filename(self) -> str:
        return Path(self.input_file).name


class DeepFilterNetProcessor:
    """DeepFilterNet speech enhancement processor."""
    
    def __init__(self):
        """Initialize DeepFilterNet model."""
        print(f"\n{'='*70}")
        print("DeepFilterNet Configuration")
        print(f"{'='*70}")
        
        # Check device
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"  Device:     CUDA ({torch.cuda.get_device_name(0)})")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = 'cpu'
            print("  Device:     CPU (will be slower)")
        
        # Initialize model
        print("\n  Loading DeepFilterNet model...")
        load_start = time.time()
        
        self.model, self.df_state, _ = init_df()
        
        load_time = time.time() - load_start
        print(f"  ✓ Model loaded in {load_time:.2f}s")
        print(f"  Sample rate: {self.df_state.sr()} Hz")
        print(f"{'='*70}\n")
    
    def process_file(self, input_path: str, output_path: str) -> ProcessingResult:
        """
        Process a single audio file through DeepFilterNet.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            
        Returns:
            ProcessingResult with timing info
        """
        target_sr = self.df_state.sr()
        
        # Load audio using librosa (more reliable)
        audio_np, orig_sr = librosa.load(input_path, sr=target_sr, mono=True)
        duration = len(audio_np) / target_sr
        
        # Convert to torch tensor with correct shape [1, samples]
        audio = torch.from_numpy(audio_np).unsqueeze(0).float()
        
        # Process
        start_time = time.time()
        enhanced = enhance(self.model, self.df_state, audio)
        processing_time = time.time() - start_time
        
        # Calculate RTF
        rtf = processing_time / duration
        
        # Save (enhanced is torch tensor)
        enhanced_np = enhanced.squeeze().cpu().numpy()
        sf.write(output_path, enhanced_np, target_sr)
        
        return ProcessingResult(
            input_file=input_path,
            output_file=output_path,
            duration_sec=duration,
            processing_time_sec=processing_time,
            rtf=rtf,
            input_sr=target_sr
        )
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> List[ProcessingResult]:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Directory with input audio files
            output_dir: Directory to save enhanced files
            
        Returns:
            List of ProcessingResult objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.webm']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(str(input_dir / ext)))
        audio_files = sorted(audio_files)
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return []
        
        print(f"Processing {len(audio_files)} files...")
        print("-" * 70)
        
        results = []
        total_duration = 0
        total_processing_time = 0
        
        for i, input_path in enumerate(audio_files):
            filename = Path(input_path).stem
            output_path = output_dir / f"{filename}_enhanced.wav"
            
            print(f"[{i+1:3d}/{len(audio_files)}] {Path(input_path).name}", end=" ")
            
            try:
                result = self.process_file(input_path, str(output_path))
                results.append(result)
                
                total_duration += result.duration_sec
                total_processing_time += result.processing_time_sec
                
                print(f"→ {result.duration_sec:.1f}s in {result.processing_time_sec:.2f}s (RTF: {result.rtf:.3f})")
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        # Summary
        if results:
            avg_rtf = total_processing_time / total_duration
            print("-" * 70)
            print(f"\nSummary:")
            print(f"  Files processed:     {len(results)}")
            print(f"  Total audio:         {total_duration:.1f}s ({total_duration/60:.1f} min)")
            print(f"  Total processing:    {total_processing_time:.1f}s ({total_processing_time/60:.1f} min)")
            print(f"  Average RTF:         {avg_rtf:.4f}x")
            print(f"  Speed:               {1/avg_rtf:.1f}x real-time")
        
        return results


def benchmark_rtf(processor: DeepFilterNetProcessor, num_runs: int = 3):
    """
    Benchmark RTF with synthetic audio of various lengths.
    """
    print(f"\n{'='*70}")
    print("RTF Benchmark (L4 GPU)")
    print(f"{'='*70}")
    
    sr = processor.df_state.sr()
    durations = [5, 10, 30, 60]  # seconds
    
    results = []
    
    for duration in durations:
        # Generate synthetic audio (white noise)
        audio = torch.randn(1, int(sr * duration))
        
        rtfs = []
        for run in range(num_runs):
            start = time.time()
            _ = enhance(processor.model, processor.df_state, audio)
            elapsed = time.time() - start
            rtfs.append(elapsed / duration)
        
        avg_rtf = np.mean(rtfs)
        std_rtf = np.std(rtfs)
        
        results.append({
            'duration': duration,
            'avg_rtf': avg_rtf,
            'std_rtf': std_rtf,
            'speed': 1/avg_rtf
        })
        
        print(f"  {duration:3d}s audio: RTF = {avg_rtf:.4f} ± {std_rtf:.4f} ({1/avg_rtf:.1f}x real-time)")
    
    # Overall average
    overall_rtf = np.mean([r['avg_rtf'] for r in results])
    print(f"\n  Overall Average RTF: {overall_rtf:.4f}x ({1/overall_rtf:.1f}x real-time)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test DeepFilterNet speech enhancement'
    )
    parser.add_argument('input', nargs='?', type=str, default=None,
                        help='Input file or directory (default: test_audio/)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: output/)')
    parser.add_argument('--benchmark', '-b', action='store_true',
                        help='Run RTF benchmark')
    parser.add_argument('--benchmark-runs', type=int, default=3,
                        help='Number of benchmark runs (default: 3)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DeepFilterNetProcessor()
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_rtf(processor, args.benchmark_runs)
        print()
    
    # Determine input/output paths
    if args.input is None:
        input_path = TEST_AUDIO_DIR
    else:
        input_path = Path(args.input)
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    # Process
    if input_path.is_file():
        # Single file
        output_path = output_dir / f"{input_path.stem}_enhanced.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {input_path}")
        result = processor.process_file(str(input_path), str(output_path))
        
        print(f"\n✓ Enhanced audio saved to: {output_path}")
        print(f"  Duration: {result.duration_sec:.1f}s")
        print(f"  Processing time: {result.processing_time_sec:.2f}s")
        print(f"  RTF: {result.rtf:.4f}x ({1/result.rtf:.1f}x real-time)")
        
    elif input_path.is_dir():
        # Directory
        results = processor.process_directory(input_path, output_dir)
        
        if results:
            print(f"\n✓ Enhanced files saved to: {output_dir}/")
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)


if __name__ == '__main__':
    main()

