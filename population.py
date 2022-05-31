import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
Log_Format = "%(asctime)s - %(message)s"
logging.basicConfig(filename = "logfile.log",
                    format = Log_Format,
                    level=logging.INFO)

from model import Model
from board import Board
import random
from math import ceil
from operator import itemgetter
import numpy as np
from pathos import multiprocessing
import time
import matplotlib.pyplot as plt
from statistics import mean
import warnings
import datetime



np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')


class Population():

    def __init__(self, population_size=40, dimensions=2, weights_init_method='zeros', players_per_game=4, hidden_layer_size=10):
        assert population_size % 2 == 0,"population_size must be divisible by 2"

        self.population_size = population_size
        self.dimensions = dimensions
        self.players_per_game = players_per_game
        self.weights_init_method = weights_init_method

        self.available_models = []
        self.models = []
        for i in range(self.population_size):
            self.models.append(Model(dimensions=[(players_per_game*4)+1,hidden_layer_size,5]))
    
    def progress(self, iterations=100, games_per_model_per_iteration=100, illegal_move_penalty=0.01, win_reward=1, total_move_reward=0.2, sigma=0.1, max_rounds=1000):
        games_per_iteration = int(games_per_model_per_iteration * self.population_size / self.players_per_game)
        illegal_move_penalty = illegal_move_penalty * -1

        for iteration in range(iterations):
            self.available_models = []
            for game in range(games_per_iteration): # play games
                player_indices = self._get_players(self.players_per_game)
                winner, turn_count, ended = self._play_MADN(player_indices=player_indices, iteration=iteration, illegal_move_penalty=illegal_move_penalty, win_reward=win_reward, total_move_reward=total_move_reward, max_rounds=max_rounds)
            
            # Select population_size/2 survivors for the next generation
            survivor_indices = [i[0] for i in sorted(enumerate([m.fitness for m in self.models]), key=lambda k: k[1], reverse=True)][:round(self.population_size/2)]
            self.models = list(itemgetter(*survivor_indices)(self.models))

            # Generate offspring by mutating each survivor
            offspring = []
            for model in self.models:
                offspring.append(self._mutate(model, sigma))
            self.models.extend(offspring)

            assert len(self.models) == self.population_size,"The number of models is no longer equal to the population size, likely because the supplied population size is uneven."

        return self.models

    def evaluate_generation(self, games_per_model=100, max_rounds=1000):
        scores = [0] * len(self.models)
        for model_index in range(len(self.models)):
            for game in range(games_per_model):
                did_model_win, turn_count = self.play_MADN_vs_random(model=self.models[model_index], max_rounds=max_rounds)
                scores[model_index] += did_model_win
        scores = [score / games_per_model for score in scores]
        return scores

    def evaluate_single_model(self, model, illegal_move_penalty, win_reward, total_move_reward, games=100, max_rounds=1000):
        wins, fitness = 0, 0
        for game in range(games):
            did_model_win, model_illegal_steps_count, move_fitness, turn_count = self.play_MADN_vs_random(model=model, max_rounds=max_rounds)
            wins += did_model_win
            fitness += did_model_win*win_reward + move_fitness*total_move_reward - model_illegal_steps_count*illegal_move_penalty
        win_fraction = wins / games
        return win_fraction, fitness

    def _mutate(self, model, sigma):

        mutated_layers = [np.random.normal(layer,scale=sigma) for layer in model.layers]
        mutated_biases = [np.random.normal(biases,scale=sigma) for biases in model.biases]
        mutated_model = Model(dimensions=model.dimensions, layers=mutated_layers, biases=mutated_biases)

        return mutated_model
    
    def _play_MADN(self, iteration, player_indices, illegal_move_penalty, win_reward, total_move_reward=0.2, max_rounds=1000):
        """_summary_

        :param int iteration: the current iteration of the population, which is required to update the fitness
        :param list player_indices: indices of the players that will play the game, referring to the self.models list
        :param int/float illegal_move_penalty: number that is subtracted from the fitness when the player makes an illegal move
        :param int/float win_reward: fitness the player receives when it wins
        :param int/float total_move_reward: total fitness reward a player can receive. If the pawns of the player moved 1/4 
            of the required boxes, the player will receive 1/4 of the reward
        :param int max_rounds: the maximum amount of rounds played per game
        :return _type_: _description_
        """
        board = Board(num_players=self.players_per_game)
        turn_index, turn_count = 0, 0
        winner = None
        move_dict = {-1: 41, -2: 42, -3: 43, -4: 44} # short dict to translate some positions to number of moved boxes

        while winner is None:

            turn_count += 1 # count the number of turns
            roll = random.randint(1, 6) # roll the die
            possible_moves = board.players[turn_index].get_possible_moves(roll) # get moves that are possible
            
            # Prepare the input
            locations = board.get_all_location(relative=True, player_num=turn_index) # get the positions of all pawns
            locations = [locations[turn_index]] + locations[:turn_index] + locations[turn_index+1:] # change the order of the positions such that the current model's pawns are in front as Lex did not think of this smh
            input = []
            for location in locations:
                for pos in location:
                    input.append(pos)
            input.append(roll)

            picked_moves = self.models[player_indices[turn_index]].pick_move(input) # get the preference of the model for each move 
            picked_moves_indices = [i[0] for i in sorted(enumerate(picked_moves), key=lambda k: k[1], reverse=True)]
            
            if possible_moves: # check if there are moves that can be executed

                # Loop over the player's preferred moves and execute the most preferred move that occurs in the moves that are possible
                for move_index in picked_moves_indices:
                    if move_index in possible_moves:
                        board.do_player_move(turn_index, move_index, roll)
                        break

                # Check if the move resulted in a win, and if so end the game
                locs = board.players[turn_index].get_pawn_locations()
                if all(pawn < 0 for pawn in locs):
                    winner = turn_index
            
            # Check if either a) the model wanted to 'pass' a turn while a move was available, or b) the model wanted to make a move
            # while no moves were available. And if so, apply a penalty to the fitness if the illegal_move_penalty parameter != 0.
            if (not possible_moves) != (picked_moves_indices[0] == 4):
                self.models[player_indices[turn_index]].update_fitness(iteration, value=illegal_move_penalty)
 
            # Check if the maximum number of rounds has been reached, and if so, perform a return
            if ceil(turn_count/len(player_indices)) >= max_rounds:
                winner = False
            
            turn_index = (turn_index + 1) % len(player_indices) # set turn index to the next player

        if winner != False:
            self.models[player_indices[winner]].update_fitness(iteration=iteration, value=win_reward) # update the fitness for the winner
            winner = True
        
        # Add the 'move reward' fitness (the maximum reward is given if all pawns are in the final boxes)
        for player_index in range(self.players_per_game):
            final_positions = board.get_all_location(relative=True, player_num=player_index)[player_index] # get the positions of all pawns for a player
            move_fitness = sum([move_dict.get(x,x) for x in final_positions]) / 170 * total_move_reward
            self.models[player_indices[player_index]].update_fitness(iteration=iteration, value=move_fitness)

        return player_indices[turn_index], turn_count, winner

    def progress_with_MP(self, iterations=100, games_per_model_per_iteration=100, illegal_move_penalty=0.01, win_reward=1, total_move_reward=0.2, sigma=0.1, max_rounds=1000, num_cores = 10):
        games_per_iteration = int(games_per_model_per_iteration * self.population_size / self.players_per_game)
        illegal_move_penalty = illegal_move_penalty * -1
        
        results_dict = []
        pool = multiprocessing.Pool(num_cores)

        start_time = time.time()

        for iteration in range(iterations):
            self.available_models = []
            player_sets = [self._get_players(self.players_per_game) for game in range(games_per_iteration)]
            jobs = []
            num_lists = num_cores
            subgames = chunks(player_sets, num_lists)



            for ID, sin_set in enumerate(subgames):
                player_models = [[self.models[player] for player in sin_game] for sin_game in sin_set]
                job = pool.apply_async(self._play_MADN_with_MP, args=(ID, player_models,sin_set, results_dict, illegal_move_penalty, win_reward, self.players_per_game, total_move_reward, max_rounds))
                jobs.append(job)
            for num, sin_job in enumerate(jobs):
                results_dict += sin_job.get()

            for match in results_dict:
                for num, _ in enumerate(match[0]):
                    self.models[match[1][num]].update_fitness(iteration, match[0][num])


            survivor_indices = [i[0] for i in sorted(enumerate([m.fitness for m in self.models]), key=lambda k: k[1], reverse=True)][:round(self.population_size/2)]
            self.models = list(itemgetter(*survivor_indices)(self.models))

            # Generate offspring by mutating each survivor  
            offspring = []
            for model in self.models:
                offspring.append(self._mutate(model, sigma))
            self.models.extend(offspring)

            assert len(self.models) == self.population_size,"The number of models is no longer equal to the population size, likely because the supplied population size is uneven."
        pool.close()

        return self.models

    def _play_MADN_with_MP(self, ID, game, player_ids, results_dict, illegal_move_penalty, win_reward, players_per_game=4, total_move_reward=0.2, max_rounds=1000):
        all_game_results = []
        for num, players in enumerate(game):


            fitness_list = [0] * players_per_game
            board = Board(num_players=players_per_game)
            turn_index, turn_count = 0, 0
            winner = None
            move_dict = {-1: 41, -2: 42, -3: 43, -4: 44} # short dict to translate some positions to number of moved boxes

            while winner is None:

                turn_count += 1 # count the number of turns
                roll = random.randint(1, 6) # roll the die
                possible_moves = board.players[turn_index].get_possible_moves(roll) # get moves that are possible

                # Prepare the input
                locations = board.get_all_location(relative=True, player_num=turn_index) # get the positions of all pawns
                locations = [locations[turn_index]] + locations[:turn_index] + locations[turn_index+1:] # change the order of the positions such that the current model's pawns are in front as Lex did not think of this smh
                input = []
                for location in locations:
                    for pos in location:
                        input.append(pos)
                input.append(roll)

                picked_moves = players[turn_index].pick_move(input) # get the preference of the model for each move
                picked_moves_indices = [i[0] for i in sorted(enumerate(picked_moves), key=lambda k: k[1], reverse=True)]

                if possible_moves: # check if there are moves that can be executed

                    # Loop over the player's preferred moves and execute the most preferred move that occurs in the moves that are possible
                    for move_index in picked_moves_indices:
                        if move_index in possible_moves:
                            board.do_player_move(turn_index, move_index, roll)
                            break


                    locs = board.players[turn_index].get_pawn_locations()
                    # # Check if the move resulted in a win, and if so end the game
                    if all(pawn < 0 for pawn in locs):
                        winner = turn_index

                    # Alternate game rules: if any one of the pawns enters the finish, the game is won by that player
                    # if any(pawn < 0 for pawn in locs):
                    #     winner = turn_index

                # Check if either a) the model wanted to 'pass' a turn while a move was available, or b) the model wanted to make a move
                # while no moves were available. And if so, apply a penalty to the fitness if the illegal_move_penalty parameter != 0.
                if (not possible_moves) != (picked_moves_indices[0] == 4):
                    fitness_list[turn_index] += illegal_move_penalty

                # Check if the maximum number of rounds has been reached
                if ceil(turn_count/players_per_game) >= max_rounds:
                    winner = False

                turn_index = (turn_index + 1) % players_per_game # set turn index to the next player

            if winner != False:
                fitness_list[winner] += win_reward # update the fitness for the winner
                winner = True

            # Add the 'move reward' fitness (the maximum reward is given if all pawns are in the final boxes)
            for player_index in range(players_per_game):
                final_positions = board.get_all_location(relative=True, player_num=player_index)[player_index] # get the positions of all pawns for a player
                move_fitness = sum([move_dict.get(x,x) for x in final_positions]) / 170 * total_move_reward
                fitness_list[player_index] += move_fitness

            all_game_results.append([fitness_list,player_ids[num] ,turn_count, winner]) # write results to multi-threading safe dict
        return all_game_results


    def play_MADN_vs_random(self, model, max_rounds=1000):
        board = Board(num_players=self.players_per_game)
        turn_index, turn_count, model_illegal_steps_count = 0, 0, 0
        model_turn_index = random.randint(0,3)
        move_dict = {-1: 41, -2: 42, -3: 43, -4: 44} # short dict to translate some positions to number of moved boxes
        winner = None

        while winner is None:

            turn_count += 1 # count the number of turns
            roll = random.randint(1, 6) # roll the die
            possible_moves = board.players[turn_index].get_possible_moves(roll) # get moves that are possible
            
            if turn_index == model_turn_index: # perform a turn of the model

                # Prepare the input
                locations = board.get_all_location(relative=True, player_num=turn_index) # get the positions of all pawns
                locations = [locations[turn_index]] + locations[:turn_index] + locations[turn_index+1:] # change the order of the positions such that the current model's pawns are in front as Lex did not think of this smh
                input = []
                for location in locations:
                    for pos in location:
                        input.append(pos)
                input.append(roll)

                picked_moves = model.pick_move(input) # get the preference of the model for each move 
                picked_moves_indices = [i[0] for i in sorted(enumerate(picked_moves), key=lambda k: k[1], reverse=True)]
                
                if possible_moves: # check if there are moves that can be executed

                    # Loop over the player's preferred moves and execute the most preferred move that occurs in the moves that are possible
                    for move_index in picked_moves_indices:
                        if move_index in possible_moves:
                            board.do_player_move(turn_index, move_index, roll)
                            break

                    locs = board.players[turn_index].get_pawn_locations()
                    # # Check if the move resulted in a win, and if so end the game
                    if all(pawn < 0 for pawn in locs):
                        winner = turn_index

                    # Alternate game rules: if any one of the pawns enters the finish, the game is won by that player
                    # if any(pawn < 0 for pawn in locs):
                    #     winner = turn_index

                # Check if either a) the model wanted to 'pass' a turn while a move was available, or b) the model wanted to make a move
                # while no moves were available. And if so, apply a penalty to the fitness if the illegal_move_penalty parameter != 0.
                if (not possible_moves) != (picked_moves_indices[0] == 4):
                    model_illegal_steps_count += 1

            else: # perform a turn by randomly doing a move
                
                if possible_moves: # check if there are moves that can be executed

                    move_index = random.randint(0, len(possible_moves)-1) # randomly decide on a move
                    board.do_player_move(turn_index, possible_moves[move_index], roll) # execute said move

                    locs = board.players[turn_index].get_pawn_locations()
                    # Check if the move resulted in a win, and if so end the game
                    if all(pawn < 0 for pawn in locs):
                        winner = turn_index

                    # Alternate game rules: if any one of the pawns enters the finish, the game is won by that player
                    # if any(pawn < 0 for pawn in locs):
                    #     winner = turn_index
 
            # Check if the maximum number of rounds has been reached, and if so, perform a return
            if ceil(turn_count/self.players_per_game) >= max_rounds:
                model_final_positions = board.get_all_location(relative=True, player_num=model_turn_index)[model_turn_index]
                move_fitness = sum([move_dict.get(x,x) for x in model_final_positions]) / 170
                return False, model_illegal_steps_count, move_fitness, turn_count
            
            turn_index = (turn_index + 1) % self.players_per_game # set turn index to the next player
        
        # Compute the number of moved boxes ratio for the model
        model_final_positions = board.get_all_location(relative=True, player_num=model_turn_index)[model_turn_index]
        move_fitness = sum([move_dict.get(x,x) for x in model_final_positions]) / 170

        did_model_win = model_turn_index == ((turn_index-1)%self.players_per_game)
        return did_model_win, model_illegal_steps_count, move_fitness, turn_count

    def _get_players(self, players_per_game=4):
        if not self.available_models:
            available_models = [i for i in range(self.population_size)]
            random.shuffle(available_models)
            self.available_models = available_models

        players = []
        for i in range(players_per_game):
            players.append(self.available_models.pop())
        return players


    def evaluate_single_model_mp(self, model, illegal_move_penalty, win_reward, total_move_reward, games=100, max_rounds=1000, num_cores=10):
        num_subgames = [games // num_cores + (1 if x < games % num_cores else 0) for x in range(num_cores)]
        pool = multiprocessing.Pool(num_cores)
        jobs = []
        for num_games in num_subgames:
            job = pool.apply_async(self.process_single_model, args=(model, illegal_move_penalty, win_reward, total_move_reward, num_games, max_rounds))
            jobs.append(job)
        results = []
        for sin_job in jobs:
            results.append(sin_job.get())
        pool.close()
        all_wins = []
        all_fitnesses = []
        for sin_result in results:
            all_wins += sin_result[0]
            all_fitnesses += sin_result[1]
        win_fraction = (sum(all_wins) / len(all_wins))
        fitness = mean(all_fitnesses)
        return win_fraction, fitness


    def process_single_model(self, model, illegal_move_penalty, win_reward, total_move_reward, games=100, max_rounds=1000):
        wins, fitness = [], []
        for game in range(games):
            did_model_win, model_illegal_steps_count, move_fitness, turn_count = self.play_MADN_vs_random(model=model, max_rounds=max_rounds)
            wins.append(did_model_win)
            fitness.append(did_model_win*win_reward + move_fitness*total_move_reward - model_illegal_steps_count*illegal_move_penalty)
        return wins, fitness

def make_plots(do_log = "print"):
    # params
    population_size = 100
    iterations = 500
    games_per_model_per_iteration = 300
    illegal_move_penalty = 0.2
    total_move_reward = 2
    win_reward = 5
    sigma = 0.5
    max_rounds = 200
    num_eval_games = 1000
    num_cores = 16
    processing_times = []
    start = time.time()
    win_rates, fitnesses = [], []

    p = Population(population_size=population_size, hidden_layer_size=40)

    for i in range(iterations):
        start_iteration = time.time()
        # Progress the population one generation at the time
        log_output(do_log, f'Gen {i} | starting at {round(time.time()-start, 5)}')

        if processing_times:
            time_till_done = datetime.timedelta(seconds=(mean(processing_times)*(iterations-i)))
        else:
            time_till_done = "undefined"
        log_output(do_log, f"Estimated to be done in {time_till_done}")
        models = p.progress_with_MP(iterations=1, games_per_model_per_iteration=games_per_model_per_iteration,
            illegal_move_penalty=illegal_move_penalty, win_reward=win_reward, total_move_reward=total_move_reward, sigma=sigma, max_rounds=max_rounds, num_cores=num_cores)
        fittest_model = models[0]

        # Do evaluation
        log_output(do_log, "Evaluating...")
        win_rate, fitness = p.evaluate_single_model_mp(model=fittest_model, illegal_move_penalty=illegal_move_penalty, win_reward=win_reward, total_move_reward=total_move_reward, games=num_eval_games, max_rounds=max_rounds,num_cores=num_cores)
        win_rates.append(win_rate)
        fitnesses.append(fitness)
        time_last_iteration = time.time() - start_iteration
        processing_times.append(time_last_iteration)
        log_output(do_log, f"Gained fitnes: {round(fitness, 5)} and winrate: {round(win_rate, 5)}")
        log_output(do_log, f"delta fitnes: {round(fitness - mean(fitnesses), 5)} | delta winrate: {round(win_rate - 0.25, 5)}")
        log_output(do_log, "-"*40)

    end = time.time()
    log_output(do_log, f"Final processing time is {end-start} seconds")

    plot_winrate(win_rates, population_size, iterations, games_per_model_per_iteration, num_eval_games, illegal_move_penalty, total_move_reward, win_reward, sigma)
    plot_fitness(fitnesses, population_size, iterations, games_per_model_per_iteration, num_eval_games, illegal_move_penalty, total_move_reward, win_reward, sigma)
    plot_fitness_incremental(fitnesses, population_size, iterations, games_per_model_per_iteration, num_eval_games, illegal_move_penalty, total_move_reward, win_reward, sigma)

def plot_winrate(win_rates, population_size, iterations, games_per_model_per_iteration, num_eval_games, illegal_move_penalty, total_move_reward, win_reward, sigma):
    plt.plot((win_rates))
    plt.title("Plot of winrates of the fittestst models of each generation against random models.")
    plt.ylabel('Win rate of fittest model vs randomized models')
    plt.axhline(y=0.25, color='r')
    plt.xlabel('Generation')
    plt.savefig(f'winrates_{population_size}_{iterations}_{games_per_model_per_iteration}.png')

def plot_fitness(fitnesses, population_size, iterations, games_per_model_per_iteration, num_eval_games, illegal_move_penalty, total_move_reward, win_reward, sigma):
    plt.plot((fitnesses))
    plt.title("Plot of fitness of the fittestst models of each generation against random models.")
    plt.ylabel('Fitness of fittest model vs randomized models')
    plt.xlabel('Generation')
    plt.savefig(f'fitnesses_{population_size}_{iterations}_{games_per_model_per_iteration}.png')

def plot_fitness_incremental(fitnesses, population_size, iterations, games_per_model_per_iteration, num_eval_games, illegal_move_penalty, total_move_reward, win_reward, sigma):
    incremental_fitnesses = [max([fitnesses[i], max(fitnesses[:i], default=0)]) for i in range(len(fitnesses))]
    plt.plot(incremental_fitnesses)
    plt.title("Plot of incremental fitness of the fittestst models of each generation against random models.")
    plt.ylabel('Incremental fitness of fittest model vs randomized models')
    plt.xlabel('Generation')
    plt.savefig(f'incremental_fitnesses_{population_size}_{iterations}_{games_per_model_per_iteration}.png')

def log_output(do_log, message):
    if do_log == "log":
        logging.info(message)
    if do_log == "print":
        print(message)
    if do_log == "both":
        print(message)
        logging.info(message)

def chunks(list_to_split, num_subs):
    """Yield n number of striped chunks from l."""
    new_list = []
    list_len = len(list_to_split)
    if list_len < num_subs:
        num_subs = list_len
    chunk_sizes = int(list_len / num_subs)
    for i in range(0, num_subs):
        new_list.append(list_to_split[i*chunk_sizes:(i+1)*chunk_sizes])
    return new_list



if __name__ == '__main__':
    make_plots(do_log="log")
