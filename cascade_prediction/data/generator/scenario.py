"""
Scenario Orchestrator Module
=============================

PURPOSE:
--------
High-level orchestration for batch scenario generation with train/val/test splitting.
This module handles the complete workflow of generating large datasets for training.

WORKFLOW:
---------
1. Generate scenarios in batches (normal, stressed, cascade)
2. Split into train/val/test sets
3. Save batches as pickle files
4. Track progress and handle retries

Author: Kraftgene AI Inc. (R&D)
Date: October 2025
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import gc

from .simulator import PhysicsBasedGridSimulator
from .utils import MemoryMonitor, save_scenarios
from .config import Settings


class ScenarioOrchestrator:
    """
    Orchestrates batch generation of cascade failure scenarios.
    
    This class handles:
    - Batch generation with progress tracking
    - Train/val/test splitting
    - Retry logic for failed scenarios
    - Memory management
    - File I/O
    """
    
    def __init__(
        self,
        simulator: PhysicsBasedGridSimulator,
        output_dir: str = 'data',
        batch_size: int = Settings.Scenario.DEFAULT_BATCH_SIZE,
        train_ratio: float = Settings.Dataset.TRAIN_RATIO,
        val_ratio: float = Settings.Dataset.VAL_RATIO,
        test_ratio: float = Settings.Dataset.TEST_RATIO
    ):
        """
        Initialize scenario orchestrator.
        
        Parameters:
        -----------
        simulator : PhysicsBasedGridSimulator
            Initialized simulator instance
        output_dir : str
            Output directory for generated data
        batch_size : int
            Number of scenarios per batch file
        train_ratio : float
            Fraction of data for training
        val_ratio : float
            Fraction of data for validation
        test_ratio : float
            Fraction of data for testing
        """
        self.simulator = simulator
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < Settings.Dataset.RATIO_TOLERANCE, \
            "Train/val/test ratios must sum to 1.0"
        
        # Create output directories
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_dataset(
        self,
        num_normal: int = Settings.Scenario.DEFAULT_NUM_NORMAL,
        num_cascade: int = Settings.Scenario.DEFAULT_NUM_CASCADE,
        num_stressed: int = Settings.Scenario.DEFAULT_NUM_STRESSED,
        sequence_length: int = Settings.Scenario.DEFAULT_SEQUENCE_LENGTH,
        start_batch: int = 0
    ) -> Dict[str, int]:
        """
        Generate complete dataset with train/val/test splits.
        
        Parameters:
        -----------
        num_normal : int
            Number of normal scenarios (stress 0.3-0.7, no failures)
        num_cascade : int
            Number of cascade scenarios (stress > 0.9, failures propagate)
        num_stressed : int
            Number of stressed scenarios (stress 0.8-0.9, no failures)
        sequence_length : int
            Number of timesteps per scenario
        start_batch : int
            Starting batch number (for parallel generation)
        
        Returns:
        --------
        stats : Dict[str, int]
            Generation statistics (counts per split)
        """
        total_scenarios = num_normal + num_cascade + num_stressed
        
        if total_scenarios == 0:
            print("No scenarios to generate. Exiting.")
            return {}
        
        # Calculate split counts
        splits = self._calculate_splits(num_normal, num_cascade, num_stressed)
        
        # Print generation plan
        self._print_generation_plan(splits, sequence_length, start_batch)
        
        # Generate each split
        train_stats = self._generate_split(
            splits['train'], self.train_dir, 'TRAIN', 
            sequence_length, start_batch
        )
        
        val_stats = self._generate_split(
            splits['val'], self.val_dir, 'VALIDATION',
            sequence_length, start_batch
        )
        
        test_stats = self._generate_split(
            splits['test'], self.test_dir, 'TEST',
            sequence_length, start_batch
        )
        
        # Print completion summary
        self._print_completion_summary(train_stats, val_stats, test_stats)
        
        return {
            'train': train_stats,
            'val': val_stats,
            'test': test_stats
        }
    
    def _calculate_splits(
        self,
        num_normal: int,
        num_cascade: int,
        num_stressed: int
    ) -> Dict[str, Dict[str, int]]:
        """Calculate scenario counts for each split."""
        # Normal scenarios
        num_train_normal = int(num_normal * self.train_ratio)
        num_val_normal = int(num_normal * self.val_ratio)
        num_test_normal = num_normal - num_train_normal - num_val_normal
        
        # Cascade scenarios
        num_train_cascade = int(num_cascade * self.train_ratio)
        num_val_cascade = int(num_cascade * self.val_ratio)
        num_test_cascade = num_cascade - num_train_cascade - num_val_cascade
        
        # Stressed scenarios
        num_train_stressed = int(num_stressed * self.train_ratio)
        num_val_stressed = int(num_stressed * self.val_ratio)
        num_test_stressed = num_stressed - num_train_stressed - num_val_stressed
        
        return {
            'train': {
                'normal': num_train_normal,
                'cascade': num_train_cascade,
                'stressed': num_train_stressed,
                'total': num_train_normal + num_train_cascade + num_train_stressed
            },
            'val': {
                'normal': num_val_normal,
                'cascade': num_val_cascade,
                'stressed': num_val_stressed,
                'total': num_val_normal + num_val_cascade + num_val_stressed
            },
            'test': {
                'normal': num_test_normal,
                'cascade': num_test_cascade,
                'stressed': num_test_stressed,
                'total': num_test_normal + num_test_cascade + num_test_stressed
            }
        }
    
    def _generate_split(
        self,
        split_counts: Dict[str, int],
        output_dir: Path,
        split_name: str,
        sequence_length: int,
        start_batch: int
    ) -> Dict[str, int]:
        """Generate scenarios for a single split (train/val/test)."""
        num_normal = split_counts['normal']
        num_cascade = split_counts['cascade']
        num_stressed = split_counts['stressed']
        total = split_counts['total']
        
        if total == 0:
            print(f"\n[{split_name}] No scenarios to generate. Skipping.")
            return {'generated': 0, 'failed': 0}
        
        print(f"\n{'='*80}")
        print(f"GENERATING {split_name} SET")
        print(f"{'='*80}")
        print(f"  Normal: {num_normal}, Stressed: {num_stressed}, Cascade: {num_cascade}")
        print(f"  Total: {total} scenarios")
        
        # Create scenario type list and shuffle
        types_to_gen = (
            ['normal'] * num_normal +
            ['cascade'] * num_cascade +
            ['stressed'] * num_stressed
        )
        np.random.shuffle(types_to_gen)
        
        # Generate scenarios
        current_batch = []
        batch_count = start_batch
        generated_count = 0
        failed_count = 0
        
        for i in range(total):
            gen_type = types_to_gen[i]
            
            print(f"\n[{split_name}] Scenario {i+1}/{total} (Type: {gen_type})")
            
            # Generate with retry logic
            scenario = self._generate_with_retry(gen_type, sequence_length)
            
            if scenario is not None:
                current_batch.append(scenario)
                generated_count += 1
            else:
                failed_count += 1
                print(f"  [FAILED] Skipping scenario after max retries")
            
            # Save batch if full or last scenario
            if len(current_batch) >= self.batch_size or i == total - 1:
                if len(current_batch) > 0:
                    batch_file = output_dir / f'scenarios_batch_{batch_count}.pkl'
                    with open(batch_file, 'wb') as f:
                        pickle.dump(current_batch, f)
                    
                    print(f"\n  [SAVED] Batch {batch_count}: {len(current_batch)} scenarios -> {batch_file}")
                    print(f"  Memory: {MemoryMonitor.get_memory_usage():.1f} MB")
                    
                    batch_count += 1
                    current_batch = []
                    gc.collect()
        
        print(f"\n[{split_name}] Complete: {generated_count} generated, {failed_count} failed")
        
        return {'generated': generated_count, 'failed': failed_count}
    
    def _generate_with_retry(
        self,
        scenario_type: str,
        sequence_length: int,
        max_retries: int = Settings.Scenario.MAX_RETRIES
    ) -> Optional[Dict]:
        """
        Generate a scenario with retry logic.
        
        Parameters:
        -----------
        scenario_type : str
            Type of scenario ('normal', 'stressed', 'cascade')
        sequence_length : int
            Number of timesteps
        max_retries : int
            Maximum retry attempts
        
        Returns:
        --------
        scenario : Dict or None
            Generated scenario or None if failed
        """
        for retry in range(max_retries):
            # Determine stress level based on type
            if scenario_type == 'cascade':
                stress_level = np.random.uniform(Settings.Scenario.CASCADE_STRESS_MIN, Settings.Scenario.CASCADE_STRESS_MAX)
            elif scenario_type == 'stressed':
                stress_level = np.random.uniform(Settings.Scenario.STRESSED_STRESS_MIN, Settings.Scenario.STRESSED_STRESS_MAX)
            else:  # normal
                stress_level = np.random.uniform(Settings.Scenario.NORMAL_STRESS_MIN, Settings.Scenario.NORMAL_STRESS_MAX)
            
            # Generate scenario
            scenario = self.simulator.generate_scenario(
                stress_level=stress_level,
                sequence_length=sequence_length
            )
            
            if scenario is None:
                if retry < max_retries - 1:
                    print(f"  [RETRY {retry+1}] Generation failed, retrying...")
                continue
            
            # Check if output matches desired type
            is_cascade = scenario['metadata']['is_cascade']
            
            if scenario_type == 'cascade' and not is_cascade:
                if retry < max_retries - 1:
                    print(f"  [RETRY {retry+1}] Wanted cascade, got normal. Increasing stress...")
                continue
            
            if scenario_type in ['normal', 'stressed'] and is_cascade:
                if retry < max_retries - 1:
                    print(f"  [RETRY {retry+1}] Wanted {scenario_type}, got cascade. Decreasing stress...")
                continue
            
            # Success!
            print(f"  [OK] Generated {scenario_type} scenario (stress={stress_level:.3f})")
            return scenario
        
        return None
    
    def _print_generation_plan(
        self,
        splits: Dict[str, Dict[str, int]],
        sequence_length: int,
        start_batch: int
    ):
        """Print generation plan summary."""
        total = sum(s['total'] for s in splits.values())
        
        print(f"\n{'='*80}")
        print(f"DATASET GENERATION PLAN")
        print(f"{'='*80}")
        print(f"  Total Scenarios: {total}")
        print(f"  Sequence Length: {sequence_length} timesteps")
        print(f"  Batch Size: {self.batch_size} scenarios/file")
        print(f"  Starting Batch: {start_batch}")
        
        for split_name, counts in splits.items():
            pct = counts['total'] / total * 100 if total > 0 else 0
            print(f"\n  {split_name.upper()} Set: {counts['total']} scenarios ({pct:.1f}%)")
            print(f"    Normal:   {counts['normal']}")
            print(f"    Stressed: {counts['stressed']}")
            print(f"    Cascade:  {counts['cascade']}")
    
    def _print_completion_summary(
        self,
        train_stats: Dict[str, int],
        val_stats: Dict[str, int],
        test_stats: Dict[str, int]
    ):
        """Print completion summary."""
        total_generated = (
            train_stats['generated'] +
            val_stats['generated'] +
            test_stats['generated']
        )
        total_failed = (
            train_stats['failed'] +
            val_stats['failed'] +
            test_stats['failed']
        )
        
        print(f"\n{'='*80}")
        print(f"DATA GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"  Total Generated: {total_generated} scenarios")
        print(f"  Total Failed: {total_failed} scenarios")
        print(f"\n  Output Directories:")
        print(f"    Train: {self.train_dir}")
        print(f"    Val:   {self.val_dir}")
        print(f"    Test:  {self.test_dir}")
        print(f"\n  You can now train the model using:")
        print(f"    python train_model_modular.py")
        print(f"{'='*80}\n")


def generate_dataset_from_config(
    num_nodes: int = Settings.Scenario.DEFAULT_NUM_NODES,
    num_normal: int = Settings.Scenario.DEFAULT_NUM_NORMAL,
    num_cascade: int = Settings.Scenario.DEFAULT_NUM_CASCADE,
    num_stressed: int = Settings.Scenario.DEFAULT_NUM_STRESSED,
    sequence_length: int = Settings.Scenario.DEFAULT_SEQUENCE_LENGTH,
    output_dir: str = 'data',
    batch_size: int = Settings.Scenario.DEFAULT_BATCH_SIZE,
    seed: int = Settings.Scenario.DEFAULT_SEED,
    topology_file: Optional[str] = None,
    start_batch: int = 0,
    video_path: Optional[str] = None
) -> Dict[str, int]:
    """
    Convenience function to generate dataset from configuration.
    
    Parameters:
    -----------
    num_nodes : int
        Number of nodes in grid
    num_normal : int
        Number of normal scenarios
    num_cascade : int
        Number of cascade scenarios
    num_stressed : int
        Number of stressed scenarios
    sequence_length : int
        Timesteps per scenario
    output_dir : str
        Output directory
    batch_size : int
        Scenarios per batch file
    seed : int
        Random seed
    topology_file : str, optional
        Path to saved topology
    start_batch : int
        Starting batch number (for parallel generation)
    
    Returns:
    --------
    stats : Dict[str, int]
        Generation statistics
    """
    # Create simulator
    print("Initializing simulator...")
    simulator = PhysicsBasedGridSimulator(
        num_nodes=num_nodes,
        seed=seed,
        topology_file=topology_file
    )
    if video_path is not None:
        print(f"\nLoading video signal from: {video_path}")
        simulator.env_generator.load_video(video_path)

    # Save topology if not provided
    if topology_file is None:
        topology_path = Path(output_dir) / 'grid_topology.pkl'
        topology_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        print(f"\nSaving topology to {topology_path}")
        topology_data = {
            'adjacency_matrix': simulator.adjacency_matrix,
            'edge_index': simulator.edge_index.numpy(),
            'positions': simulator.positions
        }
        with open(topology_path, 'wb') as f:
            pickle.dump(topology_data, f)
    
    # Create orchestrator
    orchestrator = ScenarioOrchestrator(
        simulator=simulator,
        output_dir=output_dir,
        batch_size=batch_size
    )
    
    # Generate dataset
    stats = orchestrator.generate_dataset(
        num_normal=num_normal,
        num_cascade=num_cascade,
        num_stressed=num_stressed,
        sequence_length=sequence_length,
        start_batch=start_batch
    )
    
    return stats
