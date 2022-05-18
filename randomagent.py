""" Ik heb nog geen zin om hier mooie documentatie te make

"""

import multiprocessing

import board
import numpy as np
from math import ceil, floor
import time
from statistics import mean


def start_simulation():
    random_numbers = list(np.random.uniform(size=10000))
    start_time = time.time()
    game = board.Board(num_players=4)
    for turn in range(100000):
        for player in [0, 1, 2, 3]:
            roll = ceil(random_numbers.pop()*6)
            # roll = 6
            pawns_to_move = game.players[player].get_possible_moves(roll)
            if pawns_to_move:
                move_to_chose = floor(len(pawns_to_move)*random_numbers.pop())
                to_pick = pawns_to_move[move_to_chose]
                game.do_player_move(player, to_pick, roll)
            locs = game.players[player].get_pawn_locations()
            if all(pawn < 0 for pawn in locs):
                # print(f"finished after {turn} turns")
                # print(f"finished in {time.time() - start_time} secconds")
                return game, turn, (time.time() - start_time)


def start_multiple_simulations(sample_size, run_num, return_dict):
    try:
        counter = 0
        all_turns = []
        all_durations = []
        player_scores = [0, 0, 0, 0]
        for i in range(sample_size):
            counter += 1
            game, turns, duration = start_simulation()
            all_turns.append(turns)
            all_durations.append(duration)
            for player_num, player in enumerate(game.get_all_location()):
                if all(x <0 for x in player):
                    player_scores[(player_num + counter) % 4] += 1
        genormaliseerd_winners = [wins/sample_size for wins in player_scores]
        genormaliseerd_tijd = mean(all_durations)
        genormaliseerd_turns = mean(all_turns)
        return_dict[run_num] = [genormaliseerd_winners, genormaliseerd_tijd, genormaliseerd_turns]
        print(f"Run number {run_num} done")
    except Exception as ex:
        return_dict[run_num] = [ex]


def main():
    number_runs = 1000
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    pool = multiprocessing.Pool(10)
    for i in range(number_runs):
        pool.apply_async(start_multiple_simulations, args=(1000, i, return_dict))

    pool.close()
    pool.join()

    all_turns = []
    all_durations = []
    all_wins = []

    for i in range(number_runs):
        all_turns.append(return_dict[i][2])
        all_durations.append(return_dict[i][1])
        all_wins.append(return_dict[i][0])

    print(f"mean number of turns is {mean(all_turns)}")
    print(f"mean duration is {mean(all_durations)}")
    print(f"mean winners is {[sum(x)/number_runs for x in zip(*all_wins)]}")


if __name__ == '__main__':
    main()