"""
Scientific Skills Loader for Kosmos Agents.

Loads domain-specific scientific skills from kosmos-claude-scientific-skills
repository and injects them into agent prompts based on task requirements.

Gap addressed: Gap 3 (Agent Integration & System Prompts)
Pattern source: kosmos-claude-scientific-skills (116 skill directories)

Key Insight: Instead of generic prompts, inject domain expertise into agent
prompts based on task requirements.

Architecture:
- Skill bundles: Pre-defined sets of related skills (e.g., "single_cell_analysis")
- Domain mapping: High-level domains (e.g., "biology") map to relevant bundles
- Auto-loading: Automatically select skills based on task type/domain
- Prompt injection: Format skills for inclusion in agent prompts

Performance: 116 skills loadable by domain, improves code quality
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import json

logger = logging.getLogger(__name__)


class SkillLoader:
    """
    Loads domain-specific scientific skills for agent prompts.

    Maps task types → relevant skill sets from kosmos-claude-scientific-skills.

    Example:
        loader = SkillLoader()
        skills_text = loader.load_skills_for_task(
            task_type="single_cell_analysis",
            include_examples=False
        )
        # Returns formatted skills for prompt injection
    """

    # Predefined skill bundles mapping domains to skill files
    SKILL_BUNDLES = {
        "single_cell_analysis": [
            "scanpy", "anndata", "scvi-tools", "cellxgene",
            "gseapy", "scrublet", "doubletdetection", "pegasus",
            "muon", "episcanpy", "bbknn", "harmonypy"
        ],
        "genomics_analysis": [
            "biopython", "pysam", "pydeseq2", "biomart",
            "ensembl-database", "ncbi-gene-database", "pybedtools",
            "pyvcf", "scikit-allel", "pyliftover"
        ],
        "drug_discovery": [
            "rdkit", "datamol", "deepchem", "chembl-database",
            "pubchem", "dockstring", "mordred", "molfeat",
            "chemprop", "pharmacophore-modeling"
        ],
        "proteomics": [
            "pyopenms", "matchms", "mass-spec-utils", "peptide-spectrum-match",
            "protein-structure-prediction", "pymol-visualization"
        ],
        "clinical_research": [
            "clinvar", "clinicaltrials", "omim-database",
            "phenotype-ontology", "disease-ontology", "medical-mesh"
        ],
        "imaging_analysis": [
            "napari", "scikit-image", "opencv-python", "cellpose",
            "stardist", "ilastik", "imagej-python", "bioimage-io"
        ],
        "neuroscience": [
            "mne-python", "nilearn", "nibabel", "nipype",
            "brain-imaging", "neural-data-analysis"
        ],
        "machine_learning": [
            "scikit-learn", "xgboost", "lightgbm", "tensorflow",
            "pytorch", "keras", "neural-network-training"
        ]
    }

    # Common skills that exist in scientific-skills directory
    # Note: pandas, numpy, matplotlib etc. are standard imports, not skill files
    COMMON_SKILLS = [
        "scikit-learn", "statsmodels", "polars", "dask"
    ]

    # Map high-level domains to specific skill bundles
    DOMAIN_TO_BUNDLES = {
        "biology": ["genomics_analysis", "single_cell_analysis", "proteomics"],
        "chemistry": ["drug_discovery"],
        "materials": ["machine_learning"],  # Materials science uses ML
        "neuroscience": ["neuroscience", "imaging_analysis"],
        "clinical": ["clinical_research"],
        "genomics": ["genomics_analysis"],
        # The observation unit, not the field: this is what the data pipeline
        # here (h5ad ingestion, per-source HVG selection, PPI) is built for, so
        # it gets the single-cell bundle plus the genomics and ML ones it uses.
        "single_cell": ["single_cell_analysis", "genomics_analysis", "machine_learning"],
        "drug_discovery": ["drug_discovery"],
        "proteomics": ["proteomics"],
        "imaging": ["imaging_analysis"],
    }

    def __init__(
        self,
        skills_dir: Optional[str] = None,
        auto_discover: bool = True
    ):
        """
        Initialize SkillLoader.

        Args:
            skills_dir: Path to scientific-skills directory
                       If None, auto-discovers from kosmos-claude-scientific-skills
            auto_discover: Automatically discover all available skills
        """
        # Find skills directory (only auto-discover if auto_discover is True)
        if skills_dir is None and auto_discover:
            skills_dir = self._find_skills_directory()

        self.skills_dir = Path(skills_dir) if skills_dir else None
        self.skills_cache: Dict[str, Dict] = {}

        # Discover available skills
        if auto_discover and self.skills_dir:
            self._discover_skills()

        logger.info(
            f"SkillLoader initialized with {len(self.skills_cache)} skills "
            f"from {self.skills_dir}"
        )

    def _find_skills_directory(self) -> Optional[str]:
        """
        Auto-discover scientific-skills directory.

        Looks in:
        1. kosmos-claude-scientific-skills/scientific-skills/ (relative to kosmos package)
        2. ../kosmos-claude-scientific-skills/scientific-skills/ (one level up)
        3. Environment variable KOSMOS_SKILLS_DIR

        Returns:
            Path to skills directory or None if not found
        """
        import os

        # Try environment variable first
        if 'KOSMOS_SKILLS_DIR' in os.environ:
            skills_path = Path(os.environ['KOSMOS_SKILLS_DIR'])
            if skills_path.exists():
                return str(skills_path)

        # Try relative paths from kosmos package
        # Get kosmos package directory
        kosmos_package_dir = Path(__file__).parent.parent.parent

        # Try kosmos-claude-scientific-skills in same directory as kosmos repo
        skills_path = kosmos_package_dir / "kosmos-claude-scientific-skills" / "scientific-skills"
        if skills_path.exists():
            logger.info(f"Found skills directory at {skills_path}")
            return str(skills_path)

        # Try one level up
        skills_path = kosmos_package_dir.parent / "kosmos-claude-scientific-skills" / "scientific-skills"
        if skills_path.exists():
            logger.info(f"Found skills directory at {skills_path}")
            return str(skills_path)

        logger.warning(
            "Could not find scientific-skills directory. "
            "Set KOSMOS_SKILLS_DIR environment variable or ensure "
            "kosmos-claude-scientific-skills is accessible."
        )
        return None

    def _discover_skills(self):
        """Discover all available skills in the skills directory."""
        if not self.skills_dir or not self.skills_dir.exists():
            return

        # Each skill is a folder with SKILL.md inside
        # Look for skill folders (directories that contain SKILL.md)
        skill_count = 0
        for skill_folder in self.skills_dir.iterdir():
            if skill_folder.is_dir():
                skill_file = skill_folder / "SKILL.md"
                if skill_file.exists():
                    skill_name = skill_folder.name  # Use folder name as skill name
                    self.skills_cache[skill_name] = {
                        'path': str(skill_file),
                        'name': skill_name,
                        'category': skill_folder.name,
                        'loaded': False  # Lazy load content
                    }
                    skill_count += 1

        logger.info(f"Discovered {skill_count} skill directories")

    def load_skill(self, skill_name: str) -> Optional[Dict]:
        """
        Load a single skill by name.

        Args:
            skill_name: Name of skill file (without .md extension)

        Returns:
            Dictionary with skill metadata and content, or None if not found
        """
        # Check cache first
        if skill_name in self.skills_cache:
            skill_info = self.skills_cache[skill_name]

            # Load content if not already loaded
            if not skill_info.get('loaded', False):
                try:
                    with open(skill_info['path'], 'r') as f:
                        content = f.read()
                        skill_info['content'] = content
                        skill_info['loaded'] = True

                        # Parse markdown to extract sections
                        parsed = self._parse_skill_markdown(content)
                        skill_info.update(parsed)

                except Exception as e:
                    logger.error(f"Failed to load skill {skill_name}: {e}")
                    return None

            return skill_info

        # Try to find skill file in folder structure (skill_name/SKILL.md)
        if self.skills_dir:
            skill_folder = self.skills_dir / skill_name
            skill_file = skill_folder / "SKILL.md"
            if skill_file.exists():
                return self.load_skill_from_file(skill_file, skill_name)

        # Don't spam warnings for missing common skills
        if skill_name not in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'scipy', 'jupyter-notebook']:
            logger.warning(f"Skill not found: {skill_name}")
        return None

    def load_skill_from_file(self, skill_path: Path, skill_name: Optional[str] = None) -> Dict:
        """Load skill from specific file path."""
        try:
            with open(skill_path, 'r') as f:
                content = f.read()

            # Use provided skill_name or derive from path
            if skill_name is None:
                # If file is SKILL.md, use parent folder name
                if skill_path.stem == "SKILL":
                    skill_name = skill_path.parent.name
                else:
                    skill_name = skill_path.stem

            skill_info = {
                'path': str(skill_path),
                'name': skill_name,
                'category': skill_path.parent.name,
                'content': content,
                'loaded': True
            }

            # Parse markdown
            parsed = self._parse_skill_markdown(content)
            skill_info.update(parsed)

            # Cache it
            self.skills_cache[skill_info['name']] = skill_info

            return skill_info

        except Exception as e:
            logger.error(f"Failed to load skill from {skill_path}: {e}")
            return {}

    def _parse_skill_markdown(self, content: str) -> Dict:
        """
        Parse skill markdown to extract key sections.

        Looks for:
        - Description (first paragraph)
        - Code examples (```python blocks)
        - Common functions/methods
        - Best practices

        Args:
            content: Markdown content

        Returns:
            Dictionary with parsed sections
        """
        lines = content.split('\n')
        parsed = {
            'description': '',
            'examples': [],
            'functions': {},
            'best_practices': []
        }

        # Extract first paragraph as description
        desc_lines = []
        for line in lines:
            if line.strip() and not line.startswith('#'):
                desc_lines.append(line.strip())
            elif desc_lines:
                break  # Stop at first heading after description

        parsed['description'] = ' '.join(desc_lines[:3])  # First 3 lines

        # Extract code examples (```python blocks)
        in_code_block = False
        current_code = []
        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                current_code = []
            elif line.strip().startswith('```') and in_code_block:
                in_code_block = False
                if current_code:
                    parsed['examples'].append('\n'.join(current_code))
            elif in_code_block:
                current_code.append(line)

        return parsed

    def load_skills_for_task(
        self,
        task_type: Optional[str] = None,
        libraries: Optional[List[str]] = None,
        domain: Optional[str] = None,
        include_examples: bool = False,
        include_common: bool = True
    ) -> str:
        """
        Load relevant skills for a task.

        Args:
            task_type: Type of analysis (e.g., "single_cell_analysis")
            libraries: Specific libraries needed
            domain: Domain name (alternative to task_type)
            include_examples: Include code examples in output
            include_common: Include common skills (pandas, numpy, etc.)

        Returns:
            Formatted skills text for prompt injection
        """
        skills_to_load: Set[str] = set()

        # Load task-specific bundle
        if task_type and task_type in self.SKILL_BUNDLES:
            skills_to_load.update(self.SKILL_BUNDLES[task_type])

        # Map domain to bundles using DOMAIN_TO_BUNDLES
        if domain:
            domain_lower = domain.lower()
            if domain_lower in self.DOMAIN_TO_BUNDLES:
                for bundle_name in self.DOMAIN_TO_BUNDLES[domain_lower]:
                    if bundle_name in self.SKILL_BUNDLES:
                        skills_to_load.update(self.SKILL_BUNDLES[bundle_name])
            # Also try domain directly as bundle name
            elif domain_lower in self.SKILL_BUNDLES:
                skills_to_load.update(self.SKILL_BUNDLES[domain_lower])

        # Add specific libraries
        if libraries:
            skills_to_load.update(libraries)

        # Add common skills
        if include_common:
            skills_to_load.update(self.COMMON_SKILLS[:5])  # Top 5 common

        # Load skill content
        loaded_skills = []
        for skill_name in skills_to_load:
            skill = self.load_skill(skill_name)
            if skill:
                loaded_skills.append(skill)

        # Format for prompt
        return self._format_skills_for_prompt(
            loaded_skills,
            include_examples=include_examples
        )

    def _format_skills_for_prompt(
        self,
        skills: List[Dict],
        include_examples: bool = False
    ) -> str:
        """
        Format skills as prompt injection.

        Args:
            skills: List of skill dictionaries
            include_examples: Include code examples

        Returns:
            Formatted skills text
        """
        if not skills:
            return ""

        prompt = "# Scientific Skills Available\n\n"
        prompt += "You have access to the following scientific libraries:\n\n"

        for skill in skills[:15]:  # Limit to 15 to avoid bloat
            prompt += f"## {skill['name']}\n\n"

            # Description
            if skill.get('description'):
                prompt += f"{skill['description']}\n\n"

            # Examples (if requested)
            if include_examples and skill.get('examples'):
                prompt += "**Example Usage**:\n```python\n"
                # Include first example only
                prompt += skill['examples'][0][:500]  # Limit length
                prompt += "\n```\n\n"

        prompt += "---\n\n"
        return prompt

    def get_available_bundles(self) -> List[str]:
        """Get list of available skill bundle names."""
        return list(self.SKILL_BUNDLES.keys())

    def get_bundle_skills(self, bundle_name: str) -> List[str]:
        """Get list of skills in a bundle."""
        return self.SKILL_BUNDLES.get(bundle_name, [])

    def search_skills(self, query: str) -> List[Dict]:
        """
        Search for skills matching a query.

        Args:
            query: Search term (matches name or description)

        Returns:
            List of matching skill dictionaries
        """
        query_lower = query.lower()
        matches = []

        for skill_name, skill_info in self.skills_cache.items():
            if query_lower in skill_name.lower():
                matches.append(self.load_skill(skill_name))
            elif query_lower in skill_info.get('description', '').lower():
                matches.append(self.load_skill(skill_name))

        return [m for m in matches if m is not None]

    def get_statistics(self) -> Dict:
        """Get statistics about available skills."""
        categories = {}
        for skill_info in self.skills_cache.values():
            category = skill_info.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1

        return {
            'total_skills': len(self.skills_cache),
            'skills_by_category': categories,
            'predefined_bundles': len(self.SKILL_BUNDLES),
            'skills_dir': str(self.skills_dir) if self.skills_dir else None
        }
