import gymnasium as gym
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from netris_rl.anchored_ppo import AnchoredPPO


def make_env():
    return gym.make("CartPole-v1")


@pytest.fixture
def anchor_model():
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
    model.learn(total_timesteps=128)
    env.close()
    return model


def test_anchored_ppo_trains(anchor_model):
    env = DummyVecEnv([make_env])
    model = AnchoredPPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        verbose=0,
        anchor_model=anchor_model,
        anchor_coef=0.1,
    )
    model.learn(total_timesteps=256)
    assert model.anchor_policy is not None
    for parameter in model.anchor_policy.parameters():
        assert not parameter.requires_grad
    env.close()


def test_anchor_policy_is_frozen(anchor_model):
    env = DummyVecEnv([make_env])
    model = AnchoredPPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
    assert model.anchor_policy is None

    model.set_anchor(anchor_model, coef=0.2)
    assert model.anchor_coef == 0.2
    for parameter in model.anchor_policy.parameters():
        assert not parameter.requires_grad
    env.close()
