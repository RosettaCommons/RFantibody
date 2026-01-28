"""
Central configuration module for RFantibody paths and settings.

This module provides a centralized location for all path configuration,
supporting environment variable overrides for flexibility across different
deployment environments.

Environment Variables:
    RFANTIBODY_ROOT: Root directory of the RFantibody project
    RFANTIBODY_WEIGHTS: Directory containing model weights
    RFANTIBODY_SCRIPTS: Directory containing inference scripts
"""

import os
from pathlib import Path
from typing import Dict


class PathConfig:
    """Central path configuration for RFantibody.

    All paths can be overridden via environment variables for flexibility
    across different deployment environments (local, Docker, HPC clusters).
    """

    # Base directories
    PROJECT_ROOT = Path(os.getenv('RFANTIBODY_ROOT', Path(__file__).parent.parent.parent))
    WEIGHTS_DIR = Path(os.getenv('RFANTIBODY_WEIGHTS', PROJECT_ROOT / 'weights'))
    SCRIPTS_DIR = Path(os.getenv('RFANTIBODY_SCRIPTS', PROJECT_ROOT / 'scripts'))

    # Scripts subdirectories
    EXAMPLES_DIR = SCRIPTS_DIR / 'examples'
    EXAMPLE_INPUTS = EXAMPLES_DIR / 'example_inputs'
    EXAMPLE_OUTPUTS = EXAMPLES_DIR / 'example_outputs'
    CONFIG_DIR = SCRIPTS_DIR / 'config'

    # Test directories
    TEST_DIR = PROJECT_ROOT / 'test'

    # Source directories
    SRC_DIR = PROJECT_ROOT / 'src'
    RFANTIBODY_DIR = SRC_DIR / 'rfantibody'

    @classmethod
    def get_test_paths(cls, module: str) -> Dict[str, Path]:
        """Get test paths for a specific module.

        Args:
            module: Name of the test module (e.g., 'rfdiffusion', 'proteinmpnn', 'rf2')

        Returns:
            Dictionary containing paths for inputs, outputs, references, and scripts
        """
        module_dir = cls.TEST_DIR / module
        return {
            'inputs': module_dir / 'inputs_for_test',
            'outputs': module_dir / 'example_outputs',
            'references': module_dir / 'reference_outputs',
            'scripts': module_dir / 'scripts',
            'reports': module_dir / 'reports'
        }

    @classmethod
    def get_weight_path(cls, model: str) -> Path:
        """Get path to model weights.

        Args:
            model: Name of the model ('rfdiffusion', 'proteinmpnn', 'rf2')

        Returns:
            Path to the model weights file

        Raises:
            ValueError: If model name is not recognized
        """
        weights = {
            'rfdiffusion': 'RFdiffusion_Ab.pt',
            'proteinmpnn': 'ProteinMPNN_v48_noise_0.2.pt',
            'rf2': 'RF2_ab.pt'
        }

        if model not in weights:
            raise ValueError(
                f"Unknown model '{model}'. Must be one of: {', '.join(weights.keys())}"
            )

        return cls.WEIGHTS_DIR / weights[model]

    @classmethod
    def get_inference_script(cls, module: str) -> Path:
        """Get path to inference script.

        Args:
            module: Name of the module ('rfdiffusion', 'proteinmpnn', 'rf2')

        Returns:
            Path to the inference script

        Raises:
            ValueError: If module name is not recognized
        """
        scripts = {
            'rfdiffusion': 'rfdiffusion_inference.py',
            'proteinmpnn': 'proteinmpnn_interface_design.py',
            'rf2': 'rf2_prediction.py'
        }

        if module not in scripts:
            raise ValueError(
                f"Unknown module '{module}'. Must be one of: {', '.join(scripts.keys())}"
            )

        return cls.SCRIPTS_DIR / scripts[module]

    @classmethod
    def get_config_path(cls, module: str, config_name: str = 'base') -> Path:
        """Get path to Hydra configuration file.

        Args:
            module: Name of the module ('inference', specific module name)
            config_name: Name of the config file (without .yaml extension)

        Returns:
            Path to the configuration file
        """
        return cls.CONFIG_DIR / module / f'{config_name}.yaml'

    @classmethod
    def validate_paths(cls) -> Dict[str, bool]:
        """Validate that critical paths exist.

        Returns:
            Dictionary mapping path names to existence status
        """
        paths_to_check = {
            'project_root': cls.PROJECT_ROOT,
            'weights_dir': cls.WEIGHTS_DIR,
            'scripts_dir': cls.SCRIPTS_DIR,
            'src_dir': cls.SRC_DIR,
            'test_dir': cls.TEST_DIR
        }

        return {name: path.exists() for name, path in paths_to_check.items()}

    @classmethod
    def ensure_output_dir(cls, path: Path) -> Path:
        """Ensure an output directory exists, creating it if necessary.

        Args:
            path: Path to the output directory

        Returns:
            The same path (for convenience)
        """
        path.mkdir(parents=True, exist_ok=True)
        return path


# For backwards compatibility and convenience
def get_project_root() -> Path:
    """Get the project root directory."""
    return PathConfig.PROJECT_ROOT


def get_weights_dir() -> Path:
    """Get the weights directory."""
    return PathConfig.WEIGHTS_DIR


def get_scripts_dir() -> Path:
    """Get the scripts directory."""
    return PathConfig.SCRIPTS_DIR
