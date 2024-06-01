import datasets.builder
from datasets.utils.version import Version
from dataclasses import dataclass, field

datasets.builder.BuilderConfig.__dataclass_fields__['version'].default_factory = lambda: Version("1.0.0")
