"""Tests for CooldownSchedule training utilities."""

import pytest

from event_jepa_cube.training import CooldownSchedule


class TestCooldownSchedule:
    """Tests for CooldownSchedule."""

    def test_warmup_ramps_lr(self):
        schedule = CooldownSchedule(primary_steps=1000, warmup_steps=100)
        lr_start = schedule.get_lr_multiplier(0)
        lr_mid = schedule.get_lr_multiplier(50)
        lr_end = schedule.get_lr_multiplier(100)
        assert lr_start == pytest.approx(schedule.warmup_lr_ratio)
        assert lr_start < lr_mid < lr_end

    def test_primary_phase_constant_after_warmup(self):
        schedule = CooldownSchedule(primary_steps=1000, warmup_steps=100)
        lr_200 = schedule.get_lr_multiplier(200)
        lr_500 = schedule.get_lr_multiplier(500)
        lr_999 = schedule.get_lr_multiplier(999)
        assert lr_200 == pytest.approx(1.0)
        assert lr_500 == pytest.approx(1.0)
        assert lr_999 == pytest.approx(1.0)

    def test_cooldown_phase_lr_decays(self):
        schedule = CooldownSchedule(primary_steps=1000, cooldown_steps=200)
        lr_start = schedule.get_lr_multiplier(1000)
        lr_mid = schedule.get_lr_multiplier(1100)
        lr_end = schedule.get_lr_multiplier(1200)
        assert lr_start > lr_mid > lr_end
        assert lr_end == pytest.approx(schedule.cooldown_end_lr_ratio)

    def test_cooldown_end_clamps(self):
        schedule = CooldownSchedule(primary_steps=100, cooldown_steps=50)
        lr = schedule.get_lr_multiplier(200)  # past total_steps
        assert lr == pytest.approx(schedule.cooldown_end_lr_ratio)

    def test_resolution_scale_primary(self):
        schedule = CooldownSchedule(primary_steps=1000, cooldown_resolution_scale=2.0)
        assert schedule.get_resolution_scale(0) == 1.0
        assert schedule.get_resolution_scale(999) == 1.0

    def test_resolution_scale_cooldown(self):
        schedule = CooldownSchedule(primary_steps=1000, cooldown_resolution_scale=2.0)
        assert schedule.get_resolution_scale(1000) == 2.0
        assert schedule.get_resolution_scale(1100) == 2.0

    def test_sequence_length(self):
        schedule = CooldownSchedule(primary_steps=100, cooldown_resolution_scale=2.0)
        assert schedule.get_sequence_length(16, step=50) == 16
        assert schedule.get_sequence_length(16, step=100) == 32

    def test_is_cooldown(self):
        schedule = CooldownSchedule(primary_steps=100, cooldown_steps=50)
        assert not schedule.is_cooldown(0)
        assert not schedule.is_cooldown(99)
        assert schedule.is_cooldown(100)
        assert schedule.is_cooldown(150)

    def test_total_steps(self):
        schedule = CooldownSchedule(primary_steps=1000, cooldown_steps=200)
        assert schedule.total_steps == 1200

    def test_negative_step(self):
        schedule = CooldownSchedule()
        lr = schedule.get_lr_multiplier(-1)
        assert lr == pytest.approx(schedule.warmup_lr_ratio)
