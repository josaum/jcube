"""Training utilities inspired by V-JEPA 2.1.

Provides a two-phase training schedule (primary + cooldown) and related
helpers.  No external dependencies -- uses only Python stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CooldownSchedule:
    """Two-phase training schedule: primary + cooldown.

    Inspired by V-JEPA 2.1's training recipe where a primary training phase
    (warmup-then-constant LR) is followed by a short cooldown phase with
    decaying LR and optionally increased input resolution / sequence length.

    This class computes LR multipliers -- it does not depend on any optimizer.
    Users multiply their base learning rate by the returned value.

    Args:
        primary_steps: Number of steps in the primary training phase.
        cooldown_steps: Number of steps in the cooldown phase.
        warmup_steps: Number of warmup steps at the beginning of the primary
            phase (LR ramps linearly from ``warmup_lr_ratio`` to 1.0).
        warmup_lr_ratio: Starting LR ratio during warmup.
        cooldown_start_lr_ratio: LR ratio at the start of cooldown.
        cooldown_end_lr_ratio: LR ratio at the end of cooldown.
        cooldown_resolution_scale: Multiplier for input resolution / sequence
            length during the cooldown phase (e.g. 2.0 = double resolution).
    """

    primary_steps: int = 135000
    cooldown_steps: int = 12000
    warmup_steps: int = 12000
    warmup_lr_ratio: float = 0.19
    cooldown_start_lr_ratio: float = 1.14
    cooldown_end_lr_ratio: float = 0.002
    cooldown_resolution_scale: float = 1.5

    def get_lr_multiplier(self, step: int) -> float:
        """Get LR multiplier for the given training step.

        During the primary phase the schedule is: linear warmup then constant.
        During cooldown the LR decays via a cosine schedule from
        ``cooldown_start_lr_ratio`` to ``cooldown_end_lr_ratio``.

        Returns:
            LR multiplier in the range [cooldown_end_lr_ratio, max(1.0, cooldown_start_lr_ratio)].
        """
        if step < 0:
            return self.warmup_lr_ratio

        # Primary phase
        if step < self.primary_steps:
            if step < self.warmup_steps:
                # Linear warmup
                progress = step / max(self.warmup_steps, 1)
                return self.warmup_lr_ratio + (1.0 - self.warmup_lr_ratio) * progress
            # Constant after warmup
            return 1.0

        # Cooldown phase
        cooldown_step = step - self.primary_steps
        if cooldown_step >= self.cooldown_steps:
            return self.cooldown_end_lr_ratio

        progress = cooldown_step / max(self.cooldown_steps, 1)
        # Cosine decay
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cooldown_end_lr_ratio + (self.cooldown_start_lr_ratio - self.cooldown_end_lr_ratio) * cos_decay

    def get_resolution_scale(self, step: int) -> float:
        """Get resolution scale factor for the given step.

        Returns 1.0 during primary phase, ``cooldown_resolution_scale`` during
        the cooldown phase.
        """
        if step >= self.primary_steps:
            return self.cooldown_resolution_scale
        return 1.0

    def get_sequence_length(self, base_length: int, step: int) -> int:
        """Get sequence length for the given step.

        Returns ``base_length`` during primary phase, scaled by
        ``cooldown_resolution_scale`` during cooldown.
        """
        scale = self.get_resolution_scale(step)
        return max(1, int(base_length * scale))

    def is_cooldown(self, step: int) -> bool:
        """Check if the step is in the cooldown phase."""
        return step >= self.primary_steps

    @property
    def total_steps(self) -> int:
        """Total steps across both phases."""
        return self.primary_steps + self.cooldown_steps
