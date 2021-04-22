from src.connect4.agent import Agent


class GameLog:
    """
    data model for tracking training data games
    """

    def __init__(self, agents):
        self.log = {}
        for agent in agents:
            self.log = {
                agent.get_agent_id(): {
                    'w': 0,
                    'l': 1
                }
            }

    def update(self, agent1: Agent, agent2: Agent, result):
        """
        Update game log
        """

        if result == 1:
            self.log[agent1.get_agent_id()]['w'] += 1
            self.log[agent2.get_agent_id()]['l'] += 1
        else:
            self.log[agent1.get_agent_id()]['l'] += 1
            self.log[agent2.get_agent_id()]['w'] += 1

    def add_new_agent(self, agent_id):
        """
        add new agent
        """
        self.log[agent_id] = {
            'w': 0,
            'l': 1
        }

    def get_log(self):
        """
        returns copy of log
        """

        return self.log.copy()
