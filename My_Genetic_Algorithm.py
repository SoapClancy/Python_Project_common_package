# %%
import copy
import random

import numpy as np
from sklearn.metrics import mean_absolute_error


# %%
class Chromosome:
    __genes = __genes_constraints_low = __genes_constraints_high = __genes_mutation_rate = __genes_target = None
    __fitness_function = 'mean_absolute_error'
    __aux_func = __aux_var = __mutation_para = None

    @classmethod
    def set_cls(cls, **kwargs):
        for key, val in kwargs.items():
            if key == 'genes':
                cls.__genes = val
            if key == 'genes_constraints_low':
                cls.__genes_constraints_low = val
            if key == 'genes_constraints_high':
                cls.__genes_constraints_high = val
            if key == 'genes_mutation_rate':
                cls.__genes_mutation_rate = val
            if key == 'genes_target':
                cls.__genes_target = val
            if key == 'genes_fitness_function':
                cls.__fitness_function = val
            if key == 'genes_mutation_para':
                cls.__mutation_para = val
            if key == 'aux_func':
                cls.__aux_func = val
            if key == 'aux_var':
                cls.__aux_var = val

    def __init__(self, *args):
        if args.__len__() == 0:
            self.__genes = Chromosome.genes_mutation(Chromosome.__genes)
        else:
            self.__genes = args[0]
        self.__fitness = self.cal_fitness(Chromosome.__fitness_function)

    def __str__(self):
        return 'The fitness of this chromosome = ' + str(self.__fitness)

    def get_genes(self):
        return self.__genes

    def set_genes(self, genes):
        self.__genes = genes
        self.__fitness = self.cal_fitness(Chromosome.__fitness_function)

    @staticmethod
    def genes_mutation(genes, method='Normal', **kwargs):
        genes_mutated = copy.deepcopy(genes)
        rand = np.random.uniform(0, 1, genes.size)
        mutation_idx = rand < Chromosome.__genes_mutation_rate
        disturbance = np.random.normal(loc=Chromosome.__mutation_para['loc'], scale=Chromosome.__mutation_para['scale'],
                                       size=genes.size)
        # NMB的python中的参数传递是name binding，而且[]操作是浅复制！！！
        # if mutation_idx.size == 1:
        #     if mutation_idx:
        #         genes_mutated += disturbance
        # else:
        genes_mutated[mutation_idx] += disturbance[mutation_idx]
        Chromosome.genes_constraints_modify(genes_mutated)
        return genes_mutated

    @staticmethod
    def genes_constraints_modify(genes):
        genes[genes < Chromosome.__genes_constraints_low] = Chromosome.__genes_constraints_low
        genes[genes > Chromosome.__genes_constraints_high] = Chromosome.__genes_constraints_high

    def get_fitness(self):
        return self.__fitness

    def cal_fitness(self, fitness_function):
        if fitness_function == 'mean_absolute_error':
            fitness = (-1) * mean_absolute_error(Chromosome.__genes_target,
                                                 Chromosome.__aux_func(self.__genes, Chromosome.__aux_var))
        elif fitness_function == 'max_absolute_error':
            absolute_error = np.abs(
                Chromosome.__genes_target - Chromosome.__aux_func(self.__genes, Chromosome.__aux_var))
            fitness = (-1) * absolute_error.max()
        else:
            print('Fatal error: cal_fitness')
            fitness = np.nan
        return fitness


# %%
class Population:
    def __init__(self, n):
        self.__chromosomes = [Chromosome() for _ in range(n)]
        self.sort_chromosomes()

    def __str__(self):
        return 'There are ' + str(self.__chromosomes.__len__()) + ' chromosomes in this population'

    def get_chromosomes(self):
        return self.__chromosomes

    def set_chromosomes(self, chromosomes):
        self.__chromosomes = chromosomes
        self.sort_chromosomes()

    def sort_chromosomes(self):
        self.__chromosomes.sort(key=lambda x: x.get_fitness(), reverse=True)


# %%
class GeneticAlgorithm:
    def __init__(self, *, init_genes, population_size, target, aux_func, aux_var, constraints_low=-0.01,
                 constraints_high=1.05, mutation_rate=0.2, mutation_para=None, num_elite_chromosome=2,
                 tournament_selection_size=4, generation_limit=100, acc_limit=0, fitness_function='mean_absolute_error',
                 verbose=True):
        # 初始化染色体类的属性
        Chromosome.set_cls(genes=init_genes, genes_constraints_low=constraints_low, genes_mutation_para=mutation_para,
                           genes_constraints_high=constraints_high, genes_mutation_rate=mutation_rate,
                           genes_target=target, genes_fitness_function=fitness_function, aux_func=aux_func,
                           aux_var=aux_var)
        # 初始化GA算法实例的属性
        self.__population_size = population_size
        self.__num_elite_chromosome = num_elite_chromosome
        self.__tournament_selection_size = tournament_selection_size
        self.__generation_limit = generation_limit
        self.__acc_limit = acc_limit
        self.__verbose = verbose
        # 第一次初始化不要变异基因
        self.__population = Population(self.__population_size)
        for i in range(self.__population.get_chromosomes().__len__()):
            self.__population.get_chromosomes()[i].set_genes(init_genes)
        if verbose:
            print('初始的fitness是' + str(self.get_best()['Fitness']))

    def __str__(self):
        return 'The best fitness is ' + str(self.get_best()['Fitness'])

    def get_best(self, **kwargs):
        try:
            new_fitness_function = kwargs['new_fitness_function']
        except KeyError:
            best = {'Fitness': self.__population.get_chromosomes()[0].get_fitness(),
                    'Genes': self.__population.get_chromosomes()[0].get_genes()}
        else:
            best = {'Fitness': self.__population.get_chromosomes()[0].cal_fitness(new_fitness_function),
                    'Genes': self.__population.get_chromosomes()[0].get_genes()}
        return best

    def iteration(self):
        for _ in range(self.__generation_limit):
            if self.get_best()['Fitness'] >= self.__acc_limit:
                break
            self._evolve()
            if self.__verbose:
                print('第' + str(_ + 1) + '次iteration后的fitness是' + str(self.get_best()['Fitness']))

    def _evolve(self):
        self._crossover()
        self._mutate()

    def _crossover(self):
        crossover_chromosomes = []
        for i in range(self.__num_elite_chromosome, self.__population_size):
            parent_1 = self._select_tournament_chromosomes()[0]
            parent_2 = self._select_tournament_chromosomes()[0]
            crossover_chromosomes.append(GeneticAlgorithm._crossover_chromosome(parent_1, parent_2))
        # 新的种群，由__num_elite_chromosome个精英和其它(__population_size - __num_elite_chromosome)个交配后得到的子代组成
        self.__population.set_chromosomes(self.__population.get_chromosomes()[:self.__num_elite_chromosome] +
                                          crossover_chromosomes)

    @staticmethod
    def _crossover_chromosome(parent_1, parent_2):
        rand = np.random.uniform(0, 1, parent_1.get_genes().size)
        parent_2_idx = rand < .5
        child_genes = parent_1.get_genes()
        child_genes[parent_2_idx] = parent_2.get_genes()[parent_2_idx]
        return Chromosome(child_genes)

    def _select_tournament_chromosomes(self):
        all_chromosomes = copy.deepcopy(self.__population.get_chromosomes())
        tournament_chromosomes = random.sample(all_chromosomes, self.__tournament_selection_size)
        tournament_chromosomes.sort(key=lambda x: x.get_fitness(), reverse=True)
        return tournament_chromosomes

    def _mutate(self):
        mutation_chromosomes = []
        for i in range(self.__num_elite_chromosome, self.__population_size):
            mutation_genes = Chromosome.genes_mutation(self.__population.get_chromosomes()[i].get_genes())
            mutation_chromosomes.append(Chromosome(mutation_genes))
        self.__population.set_chromosomes(self.__population.get_chromosomes()[:self.__num_elite_chromosome] +
                                          mutation_chromosomes)
