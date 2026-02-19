# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "RewardUncertaintyConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class RewardUncertaintyConfig(BaseConfig):
    """Configuration for reward-uncertainty based exploration.

    This module trains a small predictor that maps sequence-level representations
    (built from already-computed quantities such as log-probs and lengths) to
    a scalar reward prediction. The absolute prediction error is then used as
    an uncertainty signal to reshape the RL signal (by default, scaling
    advantages without changing their sign).

    Args:
        enable (bool): Whether to enable reward-uncertainty based shaping.
        mode (str): How to use the uncertainty signal. Supported:
            - "scale_advantage": multiply advantages by a positive factor
              derived from uncertainty (sign preserved).
            - "add_to_reward": add an intrinsic bonus to rewards based on
              reward prediction error magnitude. When preserve_sign is enabled,
              the bonus is clamped so the *total* shaped sequence reward keeps
              the same sign as the original.
        feature_type (str): Type of representation features used for prediction.
            Supported:
            - "token_ids" (default): use tokenized input_ids (prompt + response)
              as representation via a small token embedding + pooling.
            - "actor_hidden": use pooled last-layer hidden states from the actor
              log-prob forward pass (no token sampling / no auxiliary embedding).
            - "logprob_traj": use the full trajectory of response log-probs as
              a feature vector (one dimension per token).
            - "mean_logprob": use the mean response log-prob per sequence
              as a 1D feature.
            - "last_token_logprob": use the log-prob of the final response
              token (scalar) as feature.
            - "logprob_stats": legacy mode using length + mean/std log-prob.
        hidden_dim (int): Hidden dimension of the predictor MLP (single hidden layer).
        hidden_dims (Optional[list[int]]): Optional list of hidden layer sizes for a deeper MLP.
            If provided (non-empty), this takes precedence over hidden_dim.
        lr (float): Learning rate for the predictor optimizer.
        scale (float): Strength of uncertainty shaping (typical range 0.1–1.0).
        max_scale (float): Upper bound on multiplicative scaling factor when
            mode == "scale_advantage".
        preserve_sign (bool): When mode == "add_to_reward", whether to enforce
            that the shaped sequence reward keeps the same sign as the original.
        warmup_steps (int): Number of initial PPO steps to *disable applying* the
            uncertainty signal (no intrinsic reward / no advantage scaling). The
            predictor is still trained during warmup so it is ready afterward.
        cooldown_steps (int): Stop applying uncertainty shaping after N PPO steps
            (0 or negative = no cooldown, always apply). The predictor continues
            training but the bonus is not added to rewards.
    """

    enable: bool = False
    # Default to reward shaping (more direct, and preserves advantage pipeline invariants).
    mode: str = "add_to_reward"
    # Default to token IDs representation (prompt + response).
    feature_type: str = "token_ids"
    # Predictor MLP hidden dimension.
    hidden_dim: int = 64
    # Optional: predictor MLP hidden dimensions (multi-layer). If set and non-empty, overrides hidden_dim.
    hidden_dims: Optional[list[int]] = None
    # Token-ID embedding dimension (for feature_type="token_ids").
    token_embed_dim: int = 64
    # Number of tokens sampled per sequence for token_ids pooling (0 = use all valid tokens).
    token_sample_k: int = 256
    # Max number of sequences to use for predictor update per step (0 = use full batch).
    train_max_samples: int = 2048
    # Number of gradient steps per PPO step (default 1 for backward compatibility).
    train_n_updates: int = 1
    # Mini-batch size for each gradient step (only used if train_n_updates > 1).
    train_mini_batch_size: int = 256
    # Whether to train the token embedding (uses SparseAdam). If False, the embedding
    # stays fixed and the predictor learns on top of it.
    train_token_embed: bool = False

    # ----------------------------
    # actor_hidden feature options
    # ----------------------------
    # Pooling strategy for feature_type="actor_hidden":
    #  - "last": last valid token in the selected scope (typically EOS / last response token)
    #  - "mean": mean pooling over valid tokens in the selected scope
    actor_hidden_pooling: str = "mean"
    # Token scope for pooling when feature_type="actor_hidden":
    #  - "response": pool only over response tokens (recommended for RLHF-style rewards)
    #  - "full": pool over the full prompt+response sequence
    actor_hidden_scope: str = "full"
    # If True, return actor-hidden features to the driver in float16 to reduce transfer size.
    actor_hidden_fp16: bool = True
    lr: float = 1e-3
    scale: float = 0.5
    max_scale: float = 3.0
    preserve_sign: bool = True
    # Disable applying uncertainty shaping for the first N PPO steps (but still train the predictor).
    warmup_steps: int = 0
    # Stop applying uncertainty shaping after N PPO steps (0 or negative = no cooldown, always apply).
    cooldown_steps: int = 0
    # Device for predictor: "cpu" (default, safe), "cuda" (faster, requires driver GPU access), or "auto".
    device: str = "cpu"


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        mgpo_lambda (float): MGPO entropy-deviation regularization strength.
        lowacc_lambda (float): Low-accuracy focus strength.
        ds_gamma_pos (float): Differential Smoothing γ_p for positive rewards.
        ds_gamma_neg (float): Differential Smoothing γ_n for negative rewards.
        reward_uncertainty (RewardUncertaintyConfig): Reward-uncertainty based exploration/shaping.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    # MGPO: entropy-deviation regularization strength (lambda in w_ME = exp(-lambda * D_ME))
    mgpo_lambda: float = 3.0
    # Low-accuracy focus strength (lambda in w_lowacc = (1 - p_c)^lambda)
    lowacc_lambda: float = 5.0
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    # Differential Smoothing (DS) for GRPO/GSPO
    # A_i^DS = A_i + {-γ_p log π_old(y_i|x) if r_i>0; +γ_n log π_old(y_i|x) otherwise}
    ds_gamma_pos: float = 0.01  # γ_p
    ds_gamma_neg: float = 0.01  # γ_n

    # Reward-uncertainty based exploration/shaping.
    # Trains a small reward predictor on-the-fly and uses its prediction error
    # as an uncertainty signal to reshape rewards/advantages.
    reward_uncertainty: RewardUncertaintyConfig = field(default_factory=RewardUncertaintyConfig)
