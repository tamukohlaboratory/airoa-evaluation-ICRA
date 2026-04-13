import numpy as np
import torch

from openpi.policies import policy as _policy


class _DummyTorchModel:
    def to(self, device: str):
        self._device = device
        return self

    def eval(self):
        return self

    def sample_actions(self, device, observation, noise=None, **kwargs):
        del device, kwargs
        if noise is None:
            batch_size = observation.state.shape[0]
            return torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=observation.state.device)
        return noise.to(dtype=torch.float32)


def _make_obs() -> dict:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
        },
        "image_mask": {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "state": np.zeros((16,), dtype=np.float32),
    }


def test_select_mbr_candidate_prefers_consensus_candidate():
    candidates = [
        np.array([[0.0], [0.0]], dtype=np.float32),
        np.array([[0.1], [0.1]], dtype=np.float32),
        np.array([[10.0], [10.0]], dtype=np.float32),
    ]

    selected_idx, risks = _policy._select_mbr_candidate(candidates)

    assert selected_idx == 1
    assert risks[selected_idx] < risks[0]
    assert risks[selected_idx] < risks[2]


def test_select_mbr_candidate_uses_reference_set_when_provided():
    decision_candidates = [
        np.array([[0.0]], dtype=np.float32),
        np.array([[10.0]], dtype=np.float32),
    ]
    reference_candidates = [
        np.array([[9.0]], dtype=np.float32),
        np.array([[10.0]], dtype=np.float32),
        np.array([[11.0]], dtype=np.float32),
    ]

    selected_idx, risks = _policy._select_mbr_candidate(decision_candidates, reference_candidates)

    assert selected_idx == 1
    assert risks[selected_idx] < risks[0]


def test_select_mbr_candidate_wraps_angular_dimensions():
    decision_candidates = [
        np.array([[np.pi - 0.05]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
    ]
    reference_candidates = [
        np.array([[-np.pi + 0.05]], dtype=np.float32),
        np.array([[-np.pi + 0.02]], dtype=np.float32),
    ]

    selected_idx, risks = _policy._select_mbr_candidate(
        decision_candidates,
        reference_candidates,
        action_names=["base_theta"],
    )

    assert selected_idx == 0
    assert risks[selected_idx] < risks[1]


def test_policy_infer_uses_separate_reference_candidates():
    policy = _policy.Policy(
        _DummyTorchModel(),
        is_pytorch=True,
        use_mbr=True,
        mbr_num_candidates=2,
        mbr_num_reference_candidates=3,
    )
    noises = np.array(
        [
            [[0.0]],
            [[10.0]],
            [[9.0]],
            [[10.0]],
            [[11.0]],
        ],
        dtype=np.float32,
    )

    outputs = policy.infer(_make_obs(), noise=noises)

    np.testing.assert_allclose(outputs["actions"], noises[1])
    assert outputs["policy_timing"]["mbr_enabled"] is True
    assert outputs["policy_timing"]["mbr_num_candidates"] == 2
    assert outputs["policy_timing"]["mbr_num_reference_candidates"] == 3
    assert outputs["policy_timing"]["mbr_selected_index"] == 1


def test_policy_infer_allows_request_level_mbr_override():
    policy = _policy.Policy(
        _DummyTorchModel(),
        is_pytorch=True,
        use_mbr=True,
        mbr_num_candidates=3,
    )
    noises = np.array(
        [
            [[0.0]],
            [[0.1]],
            [[10.0]],
        ],
        dtype=np.float32,
    )
    obs = _make_obs()
    obs["use_mbr"] = False

    outputs = policy.infer(obs, noise=noises)

    np.testing.assert_allclose(outputs["actions"], noises[0])
    assert outputs["policy_timing"]["mbr_enabled"] is False
    assert outputs["policy_timing"]["mbr_num_candidates"] == 1
