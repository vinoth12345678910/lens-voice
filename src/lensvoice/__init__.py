"""LensVoice — laptop-only AI accessibility assistant for blind users.

This package is the runtime perception pipeline. It depends on clean model
interfaces (src/lensvoice/models/interfaces.py) and operates on framework-free
data contracts (src/lensvoice/models/schemas.py). Real model weights are
plugged in through adapters after training.
"""

__version__ = "0.1.0"