import numpy as np


def get_gae_returns(rewards: np.ndarray,
                    values: np.ndarray,
                    next_value: float,
                    done_masks: np.ndarray,
                    gamma: float,
                    lambda_: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляем оценку на advantage и value функции:
    - rewards - награды по сесии
    - values - значения value от критика
    - next_value - значение value от критика на финальном состоянии
    - done_mask - массив значений done среды
    """
    # 1. попробовать finite horizon estimators
    gae_values = np.append(values, [next_value])

    def not_done(i):
        return 1 - done_masks[i]

    def delta(i):
        return (
            rewards[i]
            + gamma * gae_values[i + 1] * not_done(i)
            - gae_values[i]
        )

    advantages = np.zeros_like(values)
    for i in range(advantages.shape[0] - 1, -1, -1):
        advantages[i] = delta(i)
        if i + 1 < advantages.shape[0]:
            advantages[i] += not_done(i) * advantages[i+1] * (gamma * lambda_)

    values_targets = advantages + np.array(values)

    # # Common trick that used with gae is advantage normilization (where source ???)
    # advantages /= np.linalg.norm(advantages)

    return advantages, values_targets