""" MensErgerJeNiet Player class
    Version 1.1
    Lex Bosch
    2022-05-17
"""
from player import Player, IllegalMoveError
import collections
import unittest


class Board:
    def __init__(self, num_players: int = 4, places_between_starts: int = 10, highest_roll: int = 6):
        """ Creates the board with players. The number of players, the number of spaces and the highest possible roll
            can be altered to wanted values.

        :param num_players: Numbers of players in the game. default is 4, but can be changed.
        :param places_between_starts: Number of spaces between the starts. Default is 10, but can be changed.
        :param highest_roll: Highest possible roll. Default is 6 but can be changed to simulate using a different die.
        """
        self.players = []
        self.num_players = num_players
        self.space = places_between_starts
        for i in range(num_players):
            self.players.append(Player(i, num_players=num_players, highest_roll=highest_roll))

    def get_all_location(self, relative: bool = True, player_num: int = 0) -> list:
        """ Returns a list of the locations of the pawns of all players. The relative locations of the pawns can be
            altered. Default is the view of player 0.

        :param relative: If set to False,  the locations of all pawns are from that players spawn. Default is True, but
        can be changed.
        :param player_num: Index of the player of who the relative location is taken. If relative is False, this does
         nothing.
        :return: Returns a nested list with the locations of the pawns of all players.
        """
        return [self.players[i].get_pawn_locations(relative, player_num) for i in range(self.num_players)]

    def do_player_move(self, player_num: int, pawn_number: int, roll: int):
        """ Move a pawn of a player by a specified number of spaces. If the pawn ends on the space of another players
         pawn, that pawn is hit and gets removed from the board.

        :param player_num: Index of player who moves a pawn.
        :param pawn_number: Index of the pawn to move
        :param roll: Number of spaces for the pawn to move
        :return:
        """
        self.players[player_num].do_move(pawn_number=pawn_number, roll=roll)
        locs = self.get_all_location()
        full_list = list(filter((0).__ne__, [element for sublist in locs for element in sublist]))
        if len(full_list) > len(set(full_list)):
            dup = ([x for x, y in collections.Counter(full_list).items() if y > 1][0])
            if dup > 0:
                locs[player_num] = []
                for player_num, player in enumerate(locs):
                    for pawn_num, pawn in enumerate(player):
                        if pawn == dup:
                            self.players[player_num].get_hit(pawn_num)
                            return

    def show_board(self):
        """ Highly WIP.
            Displays the board in a list and, in the future in a nice graphical view
        """
        if self.num_players == 4:
            full_output = [	["█"for i in range(11)] for i in range(11)]

            full_output = ["".join(part_output) for part_output in full_output]
            full_output = "\n".join(full_output)
            print(full_output)

        player_locations = self.get_all_location()

        output_dict = {}
        output_dict["full_board"] = ["+" for i in range(self.num_players*self.space)]
        for player_num, player in enumerate(player_locations):
            output_dict[f"player_{player_num}_end"] = ["+", "+", "+", "+"]
            for pawn in player:
                if pawn > 0:
                    output_dict["full_board"][pawn] = player_num
                if pawn < -1:
                    output_dict[f"player_{player_num}_end"][pawn+5] = player_num
        return output_dict


class TestBoard(unittest.TestCase):
    def test_player_move(self):
        b = Board()
        b.do_player_move(player_num=0, pawn_number=0, roll=6)
        b.do_player_move(player_num=1, pawn_number=0, roll=6)
        b.do_player_move(player_num=2, pawn_number=0, roll=6)
        self.assertEqual(b.get_all_location(True, 0),
                         [[1, 0, 0, 0], [11, 0, 0, 0], [21, 0, 0, 0], [0, 0, 0, 0]],
                         "Expected pawns to have entered board")

    def test_player_gethit(self):
        b = Board()
        b.do_player_move(player_num=0, pawn_number=0, roll=6)
        b.do_player_move(player_num=1, pawn_number=0, roll=6)
        b.do_player_move(player_num=0, pawn_number=0, roll=6)
        b.do_player_move(player_num=0, pawn_number=0, roll=4)
        self.assertEqual(b.get_all_location(True, 0),
                         [[11, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                         "Expected pawn of playernum 1 to be sent back to position 0"
                         )

    def perform_illegal_move(self):
        b = Board()
        b.do_player_move(player_num=0, pawn_number=0, roll=6)
        with self.assertRaises(IllegalMoveError):
            b.do_player_move(player_num=0, pawn_number=1, roll=6)


if __name__ == '__main__':
    unittest.main()
