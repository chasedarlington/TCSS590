import numpy as np
import torch
import pytest


def print_sequence_shapes(name, seq):
    """
    Print length/shape information for a list of arrays before torch.tensor().
    This catches ragged trajectory lists with better detail.
    """
    print(f"\n{name}:")
    print(f"  outer list length: {len(seq)}")

    shapes = []
    for i, item in enumerate(seq):
        arr = np.asarray(item)
        shapes.append(arr.shape)

        print(
            f"  {name}[{i}] "
            f"type={type(item).__name__}, "
            f"shape={arr.shape}, "
            f"len={len(item) if hasattr(item, '__len__') else 'no len'}"
        )

    unique_shapes = sorted(set(shapes))
    print(f"  unique shapes: {unique_shapes}")

    return shapes


def assert_same_shapes(name, seq):
    """
    Fail early with a clear message if a list contains arrays of different shapes.
    """
    shapes = print_sequence_shapes(name, seq)

    if len(set(shapes)) != 1:
        details = "\n".join(
            f"  {name}[{i}] shape={shape}"
            for i, shape in enumerate(shapes)
        )

        raise AssertionError(
            f"{name} is ragged. torch.tensor({name}) will likely fail.\n"
            f"Different shapes found:\n{details}\n\n"
            f"Likely cause: you used append() on whole trajectory arrays instead of "
            f"concatenate()/extend() on individual states."
        )


def test_states_all_detects_ragged_trajectories_before_torch_tensor():
    """
    This reproduces your likely bug:

    states_all = [
        trajectory with 5 states,
        trajectory with 6 states,
    ]

    Each individual state has 4 features, but the trajectory lengths differ.
    """
    obs_dim = 4

    traj_1_observations = np.zeros((5, obs_dim), dtype=np.float32)
    traj_2_observations = np.zeros((6, obs_dim), dtype=np.float32)

    states_all = []

    # This matches your current train_model pattern:
    states_all.append(traj_1_observations)
    states_all.append(traj_2_observations)

    with pytest.raises(AssertionError) as exc:
        assert_same_shapes("states_all", states_all)

    print(exc.value)

    assert "states_all is ragged" in str(exc.value)
    assert "append() on whole trajectory arrays" in str(exc.value)


def test_correct_fix_with_concatenate_produces_rectangular_tensor():
    """
    This shows the safer fix:

    np.concatenate(states_all, axis=0)

    It turns:
      [(5, 4), (6, 4)]

    into:
      (11, 4)
    """
    obs_dim = 4

    traj_1_observations = np.zeros((5, obs_dim), dtype=np.float32)
    traj_2_observations = np.zeros((6, obs_dim), dtype=np.float32)

    states_all = []
    states_all.append(traj_1_observations)
    states_all.append(traj_2_observations)

    states_np = np.concatenate(states_all, axis=0)
    states = torch.tensor(states_np, dtype=torch.float32)

    print("\nAfter np.concatenate:")
    print("  states_np.shape:", states_np.shape)
    print("  states.shape:", states.shape)

    assert states.shape == torch.Size([11, obs_dim])