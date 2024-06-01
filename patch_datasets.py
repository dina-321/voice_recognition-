from datasets.builder import BuilderConfig
from datasets.utils.version import Version
from dataclasses import field

# Update the BuilderConfig class to use default_factory for the version field
BuilderConfig.__dataclass_fields__['version'].default_factory = lambda: Version("1.0.0")
BuilderConfig.__dataclass_fields__['version'].default = None  # Ensure default is None
