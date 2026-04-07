"""
Artifact management module.

Handles saving and loading of evolution progress, checkpoints, and logs.
This module is integrated into the HybridNeuroevolution class for state persistence.
"""

import os
import json
import numpy as np
from typing import Dict, Any


class ArtifactManager:
    """
    Manages artifacts for evolution runs (progress, logs, checkpoints).
    
    This class provides utilities for saving/loading evolution state,
    but the main integration is done directly in HybridNeuroevolution class.
    """

    def __init__(self, artifacts_dir: str):
        """
        Initialize artifact manager.

        Args:
            artifacts_dir: Directory to store all artifacts
        """
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

        self.progress_json_path = os.path.join(artifacts_dir, 'evolution_progress.json')
        self.generation_log_path = os.path.join(artifacts_dir, 'generation_progress.txt')
        self.execution_log_path = os.path.join(artifacts_dir, 'execution_log.txt')

    @staticmethod
    def to_json_serializable(value: Any) -> Any:
        """
        Converts numpy types to JSON-serializable Python types.

        Args:
            value: Value to convert (can be dict, list, numpy type, etc.)

        Returns:
            JSON-serializable version of the value
        """
        if isinstance(value, dict):
            return {k: ArtifactManager.to_json_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ArtifactManager.to_json_serializable(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def save_progress(self, progress_data: Dict) -> bool:
        """
        Save evolution progress to JSON file.

        Args:
            progress_data: Dictionary containing evolution state

        Returns:
            True if successful, False otherwise
        """
        try:
            serializable_data = self.to_json_serializable(progress_data)
            with open(self.progress_json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2)
            return True
        except Exception as e:
            print(f"ERROR saving progress: {e}")
            return False

    def load_progress(self) -> Dict:
        """
        Load evolution progress from JSON file.

        Returns:
            Dictionary with evolution state, or empty dict if not found
        """
        if not os.path.exists(self.progress_json_path):
            return {}

        try:
            with open(self.progress_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load progress: {e}")
            return {}

    def append_generation_log(self, text: str):
        """
        Append text to generation log file.

        Args:
            text: Text to append
        """
        with open(self.generation_log_path, 'a', encoding='utf-8') as f:
            f.write(text + "\n")

    def initialize_generation_log(self):
        """Create/reset generation log file."""
        with open(self.generation_log_path, 'w', encoding='utf-8') as f:
            f.write("Generation progress log\n")

    def get_checkpoint_path(self, generation: int, genome_id: str, fitness: float) -> str:
        """
        Generate checkpoint filename based on generation, ID, and fitness.

        Args:
            generation: Current generation number
            genome_id: Genome identifier
            fitness: Fitness value

        Returns:
            Full path to checkpoint file
        """
        filename = f"best_model_gen{generation}_id{genome_id}_fitness{fitness:.2f}.pth"
        return os.path.join(self.artifacts_dir, filename)

    def cleanup_old_checkpoints(self, keep_path: str = None):
        """
        Remove old checkpoint files, optionally keeping one specific file.

        Args:
            keep_path: Path to checkpoint file to keep (optional)
        """
        if not os.path.exists(self.artifacts_dir):
            return

        for filename in os.listdir(self.artifacts_dir):
            if filename.startswith('best_model_') and filename.endswith('.pth'):
                filepath = os.path.join(self.artifacts_dir, filename)
                if keep_path and filepath == keep_path:
                    continue
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"WARNING: Could not remove old checkpoint {filepath}: {e}")
