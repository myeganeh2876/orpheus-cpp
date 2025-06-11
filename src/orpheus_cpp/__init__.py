# SPDX-FileCopyrightText: 2025-present Freddy Boulton <freddyboulton@hf-freddy.local>
#
# SPDX-License-Identifier: MIT

from .model import OrpheusCpp

try:
    from .optimized_model import OptimizedOrpheusCpp, TTSOptions
    __all__ = ["OrpheusCpp", "OptimizedOrpheusCpp", "TTSOptions"]
except ImportError:
    # Optimized version requires additional dependencies
    __all__ = ["OrpheusCpp"]
