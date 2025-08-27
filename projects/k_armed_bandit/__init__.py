import numpy as np
from yaspin import yaspin
import threading
import utils
import os
import matplotlib.pyplot as plt
import seaborn as sns

from models import ProjectAbstractClass

from k_armed_bandit.agents import EpsilonGreedyAgent, UCBAgent


class Project(ProjectAbstractClass):
    def __init__(self):
        ACTIONS_NUMBER = 10

        mean = 0
        std = 1

        self.rewards = [
            np.random.normal(mean,std,1).item() for _ in range(ACTIONS_NUMBER)
        ]
        # The actions are the array indexes
        # Each array contains a Normal mean and its standart deviation

        self.agents = [
            EpsilonGreedyAgent(ACTIONS_NUMBER, .01),
            UCBAgent(ACTIONS_NUMBER, .001)
        ]
    
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
        EPOCHS = 1000
        
        for _ in range(EPOCHS):
            for agent in self.agents:
                chosen_action = agent.choose()
            
                reward = self.get_reward(chosen_action)

                agent.think(reward, chosen_action)

    def run(self):
        """The function that will run the entire project structure
        """
        utils.clear_terminal()

        print("="*30)
        print("K-ARMED BANDIT PROBLEM")
        print("="*30)

        print("The values of each action are (Mean and Standart Deviation):")

        for i, reward in enumerate(self.rewards):
            print(f"Action {i}: {reward}")

        with yaspin(text="Training model, please wait a bit...", color="cyan") as spinner:
            # Separating the training into another thread so the loading animation
            # can play
            t = threading.Thread(target=self.train_agent)
            t.start()
            t.join()
            spinner.ok("✔️")

        print("="*30)

        self.generate_graphs()

    def generate_graphs(self):
        plt.title("MODEL'S TOTAL REWARD GROWTH OVER TIME")

        plt.xlabel("TIME")
        plt.ylabel("TOTAL REWARD")
        
        for agent in self.agents:
            plt.plot(
                [
                    i for i in range(sum(agent.steps)+1)
                ],
                agent.avg_reward_record,
                label=agent.name
            )

        plt.grid()

        plt.legend()

        plt.savefig(os.path.join(os.getcwd(), 'k_armed_bandit', 'images', 'graph.png'))
