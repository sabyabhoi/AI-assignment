import random

def flip(s, pos):
    return s[:pos] + ('1' if s[pos]=='0' else '0') + s[pos+1:]

def randomBitString(len):
    num = random.getrandbits(len)
    return format(num, '0'+str(len)+'b')

class Individual:
    def __init__(self, obj) -> None:
        
        if type(obj) == int:
            self.gene = randomBitString(obj)

        else:
            self.gene = obj

    def __str__(self) -> str:
        return self.gene

    def single_bit_flip(self):
        print("Before flip ", self)
        pos = random.randint(0, len(self.gene)-1) 
        test = flip(self.gene, pos)

        assert(len(test)==len(self.gene))
        for i in range(len(self.gene)):
            if i==pos:
                assert(self.gene[i] != test[i])
            else:
                assert(self.gene[i] == test[i])

        self.gene = test
        print("After flip at", pos, "the gene is", self)

# dummy fitness function for checking 
fitness_value = {}
def fitness(x : Individual):
    if x.gene not in fitness_value.keys():
        print("here")
        fitness_value[x.gene] = x.gene.count('0')
    return fitness_value[x.gene]
# end of fitness function

class GeneticAlgorithm:
    def __init__(self, num_gen, pop_size, gene_len, mutation_prob, fitness_fn):
        self.num_gen = num_gen
        self.pop_size = pop_size
        self.gene_len = gene_len
        self.mutation_prob = mutation_prob
        self.fitness_fn = fitness_fn

    def sort_population(self):
        self.population = sorted(self.population, key = lambda x: self.fitness_fn(x))
    
    def initialize(self):
        self.population = [ Individual(self.gene_len) for i in range(self.pop_size) ]
        self.sort_population()

    def run(self):
        for i in range(self.num_gen):
            if i==0:
                self.initialize()
            else:
                self.reproduce()
            print()
            print("The population of generation", i+1)
            self.print_population()
            print()
        
        return self.population[0]

    def print_population(self):
        for x in self.population:
            print(x)

    # perform elitist selection
    def selection(self):
        # top 10% of the population retained into the next generation
        next_generation = self.population[: int(0.1*self.pop_size)]
        
        # remaining candidates come from the top 50% of the population after sorting in increasing order of fitness 
        candidates = self.population[: int(0.5*self.pop_size)]        
        return next_generation, candidates

    # perform single point crossover
    def crossover(self, parent1, parent2):
        pos = random.randint(0, self.gene_len)
        print(pos, end=" : ")
        gene1 = parent1.gene
        gene2 = parent2.gene
        return Individual(gene1[:pos] + gene2[pos:]), Individual(gene2[:pos]+gene1[pos:]) 

    def mutate(self):
        i=0
        for x in self.population:
            prob = random.random()
            print(prob)

            if prob < self.mutation_prob:
                # perform single bit flip mutation
                print("Mutation for index", i)
                x.single_bit_flip()
            i+=1

    def reproduce(self):
        next_generation, candidates = self.selection()
        curr_size = len(next_generation)

        while curr_size < self.pop_size:
            parent1 = random.choice(candidates)
            parent2 = random.choice(candidates)
            print("Crossover b/w", parent1, parent2, end=", at pos ")
            child1, child2 = self.crossover(parent1, parent2) 
            print(child1, child2)
            next_generation.extend([child1, child2])
            curr_size+=2

        self.population = next_generation
        self.mutate()
        self.sort_population()
