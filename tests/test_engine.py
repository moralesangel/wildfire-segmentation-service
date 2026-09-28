"""Engine selection, and the promise that a stub is never mistaken for a model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from service.engine import StubEngine, load_engine


class TestEngineSelection:
    def test_a_missing_checkpoint_falls_back_rather_than_raising(self, tmp_path: Path) -> None:
        """A container that will not boot gives an operator nothing to look at.
        The service starts, serves, and says what it is serving with."""
        engine = load_engine(tmp_path / "absent.onnx")
        assert engine.name == "stub"

    def test_the_stub_is_named_so_it_cannot_pass_for_a_prediction(self) -> None:
        assert StubEngine().name == "stub"


class TestStubEngine:
    def test_it_returns_one_mask_per_frame(self) -> None:
        batch = np.random.default_rng(0).random((3, 3, 64, 64)).astype(np.float32)
        masks = StubEngine(delay_ms=0.0).infer(batch)

        assert masks.shape == (3, 64, 64)
        assert masks.dtype == np.float32

    def test_masks_stay_in_probability_range(self) -> None:
        # The client renders these directly; a value outside [0, 1] would be a
        # silent rendering bug rather than a visible failure.
        batch = np.random.default_rng(1).random((2, 3, 32, 32)).astype(np.float32)
        masks = StubEngine(delay_ms=0.0).infer(batch)

        assert masks.min() >= 0.0
        assert masks.max() <= 1.0

    def test_it_responds_to_the_image(self) -> None:
        """A constant mask would make a stubbed demo look broken in a way that
        is hard to tell from a broken model. This one tracks the input."""
        red = np.zeros((1, 3, 16, 16), dtype=np.float32)
        red[:, 0] = 1.0
        blue = np.zeros((1, 3, 16, 16), dtype=np.float32)
        blue[:, 2] = 1.0

        engine = StubEngine(delay_ms=0.0)
        assert engine.infer(red).mean() > engine.infer(blue).mean()


@pytest.mark.skipif(
    not Path("models/unet_fire.onnx").is_file(), reason="no exported checkpoint yet"
)
class TestOnnxEngine:
    """Runs only once a checkpoint exists, so the suite is green before training."""

    def test_it_loads_and_names_itself(self) -> None:
        engine = load_engine("models/unet_fire.onnx")
        assert engine.name == "onnx"

    def test_output_shape_matches_the_batch(self) -> None:
        engine = load_engine("models/unet_fire.onnx")
        batch = np.random.default_rng(0).random((2, 3, 128, 128)).astype(np.float32)
        masks = engine.infer(batch)

        assert masks.shape[0] == 2
        assert masks.min() >= 0.0 and masks.max() <= 1.0, "export should include the sigmoid"
