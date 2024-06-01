import datasets.builder
from datasets.utils.version import Version
from dataclasses import dataclass, field, fields

# Patch the BuilderConfig to set default_factory for version field
config_fields = fields(datasets.builder.BuilderConfig)
for field in config_fields:
    if field.name == 'version':
        field.default_factory = lambda: Version("1.0.0")
        break
