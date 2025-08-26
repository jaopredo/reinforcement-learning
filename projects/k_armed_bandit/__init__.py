import numpy as np
from yaspin import yaspin
import threading
import utils

from models import ProjectAbstractClass


class Agent:
    def __init__(self, actions_size: int, epsilon = 0.1):
        # If the agent isn't greedy, we'll check the not maximum valued action
        # with probability epsilon
        self.epsilon = epsilon

        # The total reward the agent has
        self.total_reward = 0

        # This array contains the history of each action taken by the agent
        self.estimatives = [
            0 for _ in range(actions_size)
        ]

        # How many times the i-th action was chosen
        self.steps = [
            2 for _ in range(actions_size)
        ]

        # Number of actions the agent can take
        self.actions_number = actions_size
    
    def think(self, reward: float, chosen_action: int):
        """This function adds the reward to the desired action in the agent memmory

        Args:
            reward (float): The reward given by the chosen_action
            chosen_action (int): The action it tooked
        """
        self.estimatives[chosen_action] += ( reward - self.estimatives[chosen_action] )/self.steps[chosen_action]

    def choose(self):
        # Check if the agent will explore or not
        explore = np.random.random(1) < self.epsilon
        
        if explore:  # If it will explore
            # Then I select a random action
            action = np.random.randint(0, self.actions_number)
            self.steps[action] += 1
            return action
        
        action = np.argmax(self.estimatives)  # Return the maximum estimative

        self.steps[action] += 1

        return action


class Project(ProjectAbstractClass):
    def __init__(self):
        ACTIONS_NUMBER = 10

        self.rewards = np.random.random_integers(-10, 10, ACTIONS_NUMBER)
        # The actions are the array indexes

        self.agent = Agent(ACTIONS_NUMBER)
    
    def get_reward(self, action: int):
        """Gets the reward desired for the given action

        Args:
            action (int): An integer representing the desired action

        Returns:
            int: The reward associeted with the passed action
        """
        return self.rewards[action]
    
    def train_agent(self):
        """Function responsible for the flowing of training the agent
        """
        EPOCHS = 1000000
        
        for _ in range(EPOCHS):
            chosen_action = self.agent.choose()
            
            reward = self.get_reward(chosen_action)

            self.agent.total_reward += reward

            self.agent.think(reward, chosen_action)

    def run(self):
        """The function that will run the entire project structure
        """
        utils.clear_terminal()

        print("="*30)
        print("K-ARMED BANDIT PROBLEM")
        print("="*30)

        print("Os valores de cada ação são:")

        for i, reward in enumerate(self.rewards):
            print(f"Ação {i}: {reward}")

        with yaspin(text="Training model, please wait a bit...", color="cyan") as spinner:
            t = threading.Thread(target=self.train_agent)
            t.start()
            t.join()
            spinner.ok("✔️")

        print("="*30)

        print("Após treinamento, o agente chegou na conclusão:")
        for i, reward in enumerate(self.agent.estimatives):
            print(f"Ação {i}: {reward}")

        print("="*10)
        print(f"A recompensa total obtida foi: {self.agent.total_reward}")
