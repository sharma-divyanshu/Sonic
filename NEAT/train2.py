import retro
import numpy as np
import cv2
import neat
import pickle
import visualize

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

xpos_end = 0


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx / 8)
        iny = int(iny / 8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        # cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:

            env.render()
            frame += 1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(ob, 20, 30)
            edges_high_thresh = cv2.Canny(edges, 60, 120)

            ob = np.reshape(edges_high_thresh, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)
            if rew == 666:
                print("LEVEL ENDED")
            if rew > 50:
                fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-58')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)
visualize.plot_stats(stats, ylog=False, view=True)
visualize.plot_species(stats, view=True)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
