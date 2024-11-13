import numpy as np
import random
from operator import attrgetter
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class Agent:

    def __init__(self, idx, n_nights, epsilon: float = 0.1):
        self.idx = idx
        self.epsilon = epsilon
        self.n_nights = n_nights
        self.fitness = 0
        self.parameters = np.zeros((self.n_nights))
        self.parameters[0] = 0

        # self.parameters = np.ones((self.n_nights))

    def choose_night(self):
        nights_array = [0 for _ in range(self.n_nights)]

        # if np.random.choice([True, False], 1, p=[1 - self.epsilon, self.epsilon]):
        #     night_idx = np.argmax(self.parameters)
        # else:
        #     night_idx = np.random.choice(np.arange(self.parameters.size))

        night_idx = np.argmax(self.parameters)

        nights_array[night_idx] = 1

        return np.array(nights_array)

    def mutate(self):
        self.parameters += np.random.normal(
            size=self.parameters.shape, loc=0.0, scale=0.001
        )


class Team:
    def __init__(self, individuals: list[Agent]):
        self.fitness = 0.0
        self.individuals = individuals


class Bar:

    def __init__(self, optimal_atten, n_nights):
        self.n_nights = n_nights
        self.optimal_atten = optimal_atten

    def calculate_reward(self, atten_per_night):
        reward = np.sum(atten_per_night * np.exp(-atten_per_night / self.optimal_atten))
        return reward

    def calculate_global_reward(self, total_atten_per_night):

        return self.calculate_reward(total_atten_per_night)

    def calculate_local_reward(self, agent_atten_per_night, total_atten_per_night):
        local_atten_per_night = agent_atten_per_night * total_atten_per_night

        return self.calculate_reward(local_atten_per_night)

    def calc_difference_reward(
        self,
        agent_atten_per_night: np.ndarray,
        total_atten_per_night: np.ndarray,
        cfact_type: str,
    ):
        # Counterfactual where agent attends no nights
        cfact_atten_per_night_wout_me = total_atten_per_night - agent_atten_per_night

        # Set chosen cfact attendance
        cfact_atten_per_night = None
        match (cfact_type):
            case "zero_counterfactual":
                cfact_atten_per_night = cfact_atten_per_night_wout_me
            case "fixed_first":
                cfact_fixed_first_atten_per_night = np.array(
                    [1 if night == 0 else 0 for night in range(self.n_nights)]
                )
                cfact_atten_per_night = (
                    cfact_atten_per_night_wout_me + cfact_fixed_first_atten_per_night
                )
            case "fixed_last":
                cfact_fixed_last_atten_per_night = np.array(
                    [
                        1 if night == (self.n_nights - 1) else 0
                        for night in range(self.n_nights)
                    ]
                )
                cfact_atten_per_night = (
                    cfact_atten_per_night_wout_me + cfact_fixed_last_atten_per_night
                )

        return self.calculate_reward(total_atten_per_night) - self.calculate_reward(
            cfact_atten_per_night
        )


def shuffle(population):
    for subpop in population:
        random.shuffle(subpop)


def form_teams(population, n_teams):
    teams = []

    for n_team in range(n_teams):
        team = []

        for subpop in population:
            team.append(subpop[n_team])

        teams.append(Team(team))

    return teams


def softmaxSelection(
    individuals,
    k: int,
):

    chosen_ones = []

    individuals_fitnesses = [individual.fitness for individual in individuals]
    softmax_over_fitnesses = F.softmax(torch.Tensor(individuals_fitnesses))
    selected_indexes = torch.multinomial(softmax_over_fitnesses, num_samples=k)

    for idx in selected_indexes:
        chosen_ones.append(individuals[idx])

    return chosen_ones


def epsilonGreedySelection(
    individuals,
    k: int,
    epsilon: float,
    fit_attr: str = "fitness",
):

    chosen_ones = []

    for _ in range(k):
        if np.random.choice([True, False], 1, p=[1 - epsilon, epsilon]):
            chosen_one = max(individuals, key=attrgetter(fit_attr))
        else:
            chosen_one = random.choice(individuals)

        chosen_ones.append(chosen_one)

    return sorted(chosen_ones, key=attrgetter(fit_attr), reverse=True)


def selectSubPopulation(subpopulation):

    chosen_ones = epsilonGreedySelection(
        subpopulation, len(subpopulation) // 2, epsilon=0.3
    )

    # chosen_ones = softmaxSelection(subpopulation, len(subpopulation) // 2)

    offspring = chosen_ones + chosen_ones

    # Return a deepcopy so that modifying an individual that was selected does not modify every single individual
    # that came from the same selected individual
    return [deepcopy(individual) for individual in offspring]


def select(population):
    # Perform a selection on that subpopulation and add it to the offspring population
    return [selectSubPopulation(subpop) for subpop in population]


def mutate(population, n_mutants):
    # Don't mutate the elites
    for n_individual in range(n_mutants):

        mutant_idx = n_individual + n_mutants

        for subpop in population:
            subpop[mutant_idx].mutate()
            subpop[mutant_idx].fitness = 0.0


def setPopulation(population, offspring):
    for subpop, subpop_offspring in zip(population, offspring):
        subpop[:] = subpop_offspring


def ccea(
    fitness_shaping,
    weeks,
    optimal_atten,
    n_nights,
    n_agents,
    subpopulation_size,
):
    performance = []
    bar = Bar(optimal_atten, n_nights)
    population = [
        [Agent(i, n_nights) for _ in range(subpopulation_size)] for i in range(n_agents)
    ]

    hof_team = form_teams(population, 1)[0]
    best_team = None

    for _ in range(weeks):
        # Perform selection
        offspring = select(population)
        mutate(offspring, subpopulation_size // 2)
        shuffle(offspring)
        teams = form_teams(offspring, subpopulation_size)

        for team in teams:
            selected_nights = np.array(
                [agent.choose_night() for agent in team.individuals]
            )
            total_atten_per_night = np.sum(selected_nights, axis=0)
            G = bar.calculate_global_reward(total_atten_per_night)
            team.fitness = G
            for i, agent in enumerate(team.individuals):
                match (fitness_shaping):
                    case "global":
                        agent.fitness = G
                    case "local":
                        agent.fitness = bar.calculate_local_reward(
                            selected_nights[i], total_atten_per_night
                        )
                    case "difference_fixed_first":
                        agent.fitness = bar.calc_difference_reward(
                            selected_nights[i],
                            total_atten_per_night,
                            cfact_type="fixed_first",
                        )
                    case "difference_fixed_last":
                        agent.fitness = bar.calc_difference_reward(
                            selected_nights[i],
                            total_atten_per_night,
                            cfact_type="fixed_last",
                        )
                    case "difference_zero":
                        agent.fitness = bar.calc_difference_reward(
                            selected_nights[i],
                            total_atten_per_night,
                            cfact_type="zero_counterfactual",
                        )

                    case "local_difference":
                        agent.fitness = bar.calculate_local_reward(
                            selected_nights[i], total_atten_per_night
                        ) - bar.calculate_local_reward(
                            selected_nights[i], total_atten_per_night - 1
                        )

        setPopulation(population, offspring)

        best_team = max(teams, key=lambda item: item.fitness)

        performance.append(best_team.fitness)

        if hof_team.fitness < best_team.fitness:
            hof_team = best_team

    return performance, best_team


if __name__ == "__main__":
    fitness_shaping_types = [
        "local_difference",
    ]
    weeks = 30
    subpopulation_size = 10

    n_agents = 50
    optimal_atten = 4
    n_nights = 6

    performance_dict = {}
    best_team_attendance_per_night = {}

    for fitness_shaping in fitness_shaping_types:
        performance, best_team = ccea(
            fitness_shaping,
            weeks,
            optimal_atten,
            n_nights,
            n_agents,
            subpopulation_size,
        )

        performance_dict[fitness_shaping] = performance

        selected_nights = np.array(
            [agent.choose_night() for agent in best_team.individuals]
        )
        best_team_attendance_per_night[fitness_shaping] = np.sum(
            selected_nights, axis=0
        )

    print(best_team_attendance_per_night)
    print(performance_dict)
