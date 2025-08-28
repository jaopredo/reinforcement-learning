import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def think(self, reward: float, chosen_action: int):...

    @abstractmethod
    def choose(self):...

    @abstractmethod
    def get_times(self):...
    
    @abstractmethod
    def remember_action(self, action: int):...


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
        
        # History of the actions taken
        self.actions_record = []
    
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

        self.remember_action(action)
        
        return action

    def get_times(self):
        return [ i for i in range(sum(self.steps)+1) ]

    def remember_action(self, action):
        self.actions_record.append(action)


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

        self.remember_action(action)

        return action


class GradientAgent(Agent):
    def __init__(self, actions_size: int, alpha: float = 0.1):
        # Numeric preferences for each action
        self.numeric_preferences = np.array([
            0 for _ in range(actions_size)
        ], dtype=np.float64)
        
        # Record of all rewrard averages
        self.avg_reward_record = []

        # The reward baseline for the gradient
        self.rewards_baseline = 0
        # Wich step we the agent is right now
        self.step = 0

        # History of the actions taken by the agent
        self.actions_record = []

        # The step-size parameter
        self.alpha = alpha

        self.name = "Gradient"
    
    def choose(self):
        # Make the array of probabilities
        probabilities = np.array([
            np.exp(preference) for preference in self.numeric_preferences
        ]) / (np.exp(self.numeric_preferences).sum())

        # Make the array of actions [0, 1, 2, ..., n-1]
        actions = np.arange(len(probabilities))

        # Gets the chosen action based on the probabilities
        sample = np.random.choice(actions, p=probabilities)

        self.remember_action(sample)

        return sample
    
    def think(self, reward, chosen_action):
        # Gets the probability of the chosen action
        prob_of_chosen_action = np.exp( self.numeric_preferences[chosen_action] ) / np.exp( self.numeric_preferences ).sum()

        # Updates the numeric preference of the chosen action
        self.numeric_preferences[chosen_action] += self.alpha * ( reward - self.rewards_baseline ) * (1 - prob_of_chosen_action)

        # Updates the numeric preference of the other actions that were not taken
        self.numeric_preferences[ self.numeric_preferences != chosen_action ] -= self.alpha * ( reward - self.rewards_baseline ) * prob_of_chosen_action
        
        # Updates the step
        self.step += 1

        # Updates the reward average
        self.rewards_baseline = 1/self.step * ( reward + (self.step - 1) * self.rewards_baseline )

        # Record the new reward average
        self.avg_reward_record.append(self.rewards_baseline)
    
    def get_times(self):
        return [ i for i in range(self.step) ]

    def remember_action(self, action):
        self.actions_record.append(action)
