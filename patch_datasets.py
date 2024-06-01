# patch_datasets.py

from datasets.builder import BuilderConfig
from datasets.utils.version import Version
from dataclasses import field

# Ensure the version field is correctly set up with a default factory
BuilderConfig.__dataclass_fields__['version'] = field(default_factory=lambda: Version("1.0.0"))
