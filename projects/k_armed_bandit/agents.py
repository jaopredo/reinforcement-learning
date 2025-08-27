import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def think(self):...

    @abstractmethod
    def choose(self):...


class EpsilonGreedyAgent(Agent):
    def __init__(self, actions_size: int, epsilon = 0.1):
        # If the agent isn't greedy, we'll check the not maximum valued action
        # with probability epsilon
        self.epsilon = epsilon

        # The total reward the agent has
        self.total_reward = 0
        self.avg_reward_record = [0]

        # This array contains the history of each action taken by the agent
        self.estimatives = np.array([
            2 for _ in range(actions_size)
        ])

        # How many times the i-th action was chosen
        self.steps = np.array([
            0 for _ in range(actions_size)
        ])

        # Number of actions the agent can take
        self.actions_number = actions_size

        self.name = "Epsilon Greedy"
    
    def think(self, reward: float, chosen_action: int):
        """This function adds the reward to the desired action in the agent memmory

        Args:
            reward (float): The reward given by the chosen_action
            chosen_action (int): The action it tooked
        """
        self.total_reward += reward
        self.steps[chosen_action] += 1
        self.avg_reward_record.append(self.total_reward/sum(self.steps))
        self.estimatives[chosen_action] += ( reward - self.estimatives[chosen_action] )/self.steps[chosen_action]

    def choose(self):
        # Check if the agent will explore or not
        explore = np.random.random(1) < self.epsilon
        
        if explore:  # If it will explore
            # Then I select a random action
            action = np.random.randint(0, self.actions_number)
        else:
            action = np.argmax(self.estimatives)  # Return the maximum estimative
        
        return action


class UCBAgent(EpsilonGreedyAgent):
    def __init__(self, actions_size, c):
        super().__init__(actions_size, 0)  # Epsilon doesn't matter for me in this agent
        self.constant = c

        self.name = "Upper-Conficence-Bound"

    def choose(self):
        np.seterr(divide='ignore', invalid='ignore')

        actual_step = self.steps.sum()

        action = np.argmax(
            self.estimatives + self.constant * np.sqrt(
                np.ones(self.estimatives.shape) * actual_step / self.steps
            )
        )

        return action

