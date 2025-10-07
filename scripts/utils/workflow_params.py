#!/usr/bin/env python3
"""
Utility to parameterize ComfyUI workflows and replace hardcoded paths.

This module provides functionality to load ComfyUI workflow JSON files,
replace hardcoded paths with configurable placeholders, and substitute
them with actual values at runtime.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


class WorkflowParameterizer:
    """Handle parameterization of ComfyUI workflows."""

    # Common path placeholders that should be parameterized
    PATH_PATTERNS = [
        (r'"([A-Z]:\\[^"]+)"', 'absolute_windows_path'),  # Windows absolute paths
        (r'"(/[^"]+)"', 'absolute_unix_path'),            # Unix absolute paths
        (r'"(models/[^"]+)"', 'model_path'),              # Model paths
        (r'"(checkpoints/[^"]+)"', 'checkpoint_path'),    # Checkpoint paths
        (r'"(loras/[^"]+)"', 'lora_path'),                # LoRA paths
        (r'"(embeddings/[^"]+)"', 'embedding_path'),      # Embedding paths
    ]

    # Placeholder format: {{VARIABLE_NAME}}
    PLACEHOLDER_PATTERN = r'\{\{([A-Z_]+)\}\}'

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the WorkflowParameterizer.

        Args:
            base_dir: Base directory for resolving relative paths (default: project root)
        """
        if base_dir is None:
            # Default to project root (two levels up from this file)
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)

    def parameterize_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace hardcoded paths in a workflow with placeholders.

        Args:
            workflow: ComfyUI workflow dictionary

        Returns:
            Workflow with parameterized paths
        """
        workflow_str = json.dumps(workflow, indent=2)

        # Track replacements for reporting
        replacements = {}

        for pattern, path_type in self.PATH_PATTERNS:
            matches = re.finditer(pattern, workflow_str)
            for match in matches:
                original_path = match.group(1)

                # Convert to relative path if it's within our project
                try:
                    path_obj = Path(original_path)
                    if path_obj.is_absolute():
                        try:
                            rel_path = path_obj.relative_to(self.base_dir)
                            placeholder_name = self._path_to_placeholder(rel_path)
                            placeholder = f'{{{{{placeholder_name}}}}}'
                            workflow_str = workflow_str.replace(
                                f'"{original_path}"',
                                f'"{placeholder}"'
                            )
                            replacements[original_path] = placeholder_name
                        except ValueError:
                            # Path is outside project, use generic placeholder
                            placeholder_name = f'{path_type.upper()}_{len(replacements)}'
                            placeholder = f'{{{{{placeholder_name}}}}}'
                            workflow_str = workflow_str.replace(
                                f'"{original_path}"',
                                f'"{placeholder}"'
                            )
                            replacements[original_path] = placeholder_name
                except Exception:
                    # Invalid path, skip
                    continue

        return json.loads(workflow_str), replacements

    def substitute_parameters(
        self,
        workflow: Dict[str, Any],
        parameters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Substitute placeholders in a workflow with actual values.

        Args:
            workflow: ComfyUI workflow with placeholders
            parameters: Dictionary mapping placeholder names to values.
                       If None, uses environment variables and defaults.

        Returns:
            Workflow with substituted values
        """
        if parameters is None:
            parameters = self._get_default_parameters()

        workflow_str = json.dumps(workflow)

        # Find all placeholders
        placeholders = re.findall(self.PLACEHOLDER_PATTERN, workflow_str)

        for placeholder in set(placeholders):
            if placeholder in parameters:
                value = parameters[placeholder]
                # Convert to absolute path if it's a relative path
                if not Path(value).is_absolute():
                    value = str(self.base_dir / value)

                # Normalize path separators for current OS
                value = str(Path(value))

                workflow_str = workflow_str.replace(
                    f'{{{{{placeholder}}}}}',
                    value
                )
            else:
                raise ValueError(f"Missing parameter: {placeholder}")

        return json.loads(workflow_str)

    def _path_to_placeholder(self, path: Path) -> str:
        """
        Convert a path to a placeholder name.

        Args:
            path: Path object

        Returns:
            Placeholder name in UPPER_SNAKE_CASE
        """
        # Convert path to uppercase snake case
        parts = path.parts
        placeholder = '_'.join(parts).upper()
        # Replace special characters
        placeholder = re.sub(r'[^A-Z0-9_]', '_', placeholder)
        # Remove consecutive underscores
        placeholder = re.sub(r'_+', '_', placeholder)
        return placeholder.strip('_')

    def _get_default_parameters(self) -> Dict[str, str]:
        """
        Get default parameter values from environment variables and config.

        Returns:
            Dictionary of parameter values
        """
        params = {
            # Model paths
            'MODELS_WAN2_2_WAN_2_2_BASE_SAFETENSORS': os.getenv(
                'WAN_MODEL_PATH',
                str(self.base_dir / 'models' / 'wan2.2' / 'wan_2.2_base.safetensors')
            ),
            'MODELS_SADTALKER': os.getenv(
                'SADTALKER_CHECKPOINT_PATH',
                str(self.base_dir / 'models' / 'sadtalker')
            ),
            'MODELS_WAV2LIP': os.getenv(
                'WAV2LIP_CHECKPOINT_PATH',
                str(self.base_dir / 'models' / 'wav2lip')
            ),
        }

        # Add any additional environment-based parameters
        for key, value in os.environ.items():
            if key.startswith('COMFYUI_'):
                param_name = key.replace('COMFYUI_', '')
                params[param_name] = value

        return params

    def load_and_parameterize(self, workflow_path: str) -> tuple[Dict[str, Any], Dict[str, str]]:
        """
        Load a workflow file and parameterize it.

        Args:
            workflow_path: Path to workflow JSON file

        Returns:
            Tuple of (parameterized_workflow, replacements_dict)
        """
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        return self.parameterize_workflow(workflow)

    def load_and_substitute(
        self,
        workflow_path: str,
        parameters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Load a workflow file and substitute parameters.

        Args:
            workflow_path: Path to workflow JSON file
            parameters: Parameter values (uses defaults if None)

        Returns:
            Workflow with substituted values
        """
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        return self.substitute_parameters(workflow, parameters)

    def save_workflow(self, workflow: Dict[str, Any], output_path: str):
        """
        Save a workflow to a JSON file.

        Args:
            workflow: Workflow dictionary
            output_path: Path to save the file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)


def main():
    """CLI interface for workflow parameterization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parameterize ComfyUI workflows to remove hardcoded paths"
    )

    parser.add_argument(
        'workflow',
        help="Path to workflow JSON file"
    )

    parser.add_argument(
        '--parameterize',
        action='store_true',
        help="Parameterize workflow (replace hardcoded paths with placeholders)"
    )

    parser.add_argument(
        '--substitute',
        action='store_true',
        help="Substitute parameters (replace placeholders with actual paths)"
    )

    parser.add_argument(
        '--output',
        help="Output file path (default: print to stdout)"
    )

    parser.add_argument(
        '--show-replacements',
        action='store_true',
        help="Show what paths were replaced (with --parameterize)"
    )

    args = parser.parse_args()

    parameterizer = WorkflowParameterizer()

    if args.parameterize:
        workflow, replacements = parameterizer.load_and_parameterize(args.workflow)

        if args.show_replacements:
            print("\nReplacements made:")
            print("-" * 80)
            for original, placeholder in replacements.items():
                print(f"{original}\n  -> {{{{{placeholder}}}}}\n")

        if args.output:
            parameterizer.save_workflow(workflow, args.output)
            print(f"\nParameterized workflow saved to: {args.output}")
        else:
            print(json.dumps(workflow, indent=2))

    elif args.substitute:
        workflow = parameterizer.load_and_substitute(args.workflow)

        if args.output:
            parameterizer.save_workflow(workflow, args.output)
            print(f"\nSubstituted workflow saved to: {args.output}")
        else:
            print(json.dumps(workflow, indent=2))

    else:
        parser.print_help()
        print("\nError: Please specify either --parameterize or --substitute")


if __name__ == '__main__':
    main()
