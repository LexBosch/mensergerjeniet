""" MensErgerJeNiet Player class
    Version 1.1
    Lex Bosch
    2022-05-17
"""
import unittest


class Player:
    """
    A class to represent a player in 'Mens erger je niet'

    ...

    Attributes
    ----------
    amount_players : int
        Number of players in the game
    places_between_starts : int
        Number of places that are in between the starts of the board
    pawn_location : list
        list containing the locations of the pawn of the player. Location 0 means the pawn is not yet in play and
        negative locations indicate the pawns to be in the end.
    board_size : int
        Number of tiles from the start of the baord to the end of the board.
    player_num : int
        Index of the player in the game. Player with index 0 goes first and player with higher value is later.
    highest_roll : int
        Highest roll possible. Can be altered to simulate different dice used.

    """

    def __init__(self, player_num: int, num_players: int = 4, places_between_starts: int = 10, highest_roll: int = 6):
        """ Starts player object. The parameters num_players and places_between_starts are optional and can be altered
            to create and bigger playing field with more or less players.

        :param player_num: Index of the player.
        :param num_players: Amount of players in the current game.
        :param highest_roll: highest roll possible.
        :param places_between_starts: Amount of spaces between the starts on the board.
        """
        self.amount_players = num_players
        self.places_between_starts = places_between_starts
        self.pawn_locations = [0, 0, 0, 0]
        self.board_size = (num_players * places_between_starts)
        self.player_num = player_num
        self.highest_roll = highest_roll

    def get_possible_moves(self, roll) -> list:
        """ Generates a list of all possible moves for the player given a roll. This function does not allow players
            to move their pawns if they are in the end spaces.

        :param roll: The amount of spaces the player wants to move.
        :return: List containing the indexes of pawns with possibilities to move. List can be empty if none of the
                pawns are able to move
        """
        possibilities = []

        # Section to see if there are pawns able to be played onto the board on a 6
        if roll == self.highest_roll:
            if 0 in self.pawn_locations:
                if 1 not in self.pawn_locations:
                    possibilities += [i for i, val in enumerate(self.pawn_locations) if val == 0]

        # Section to see if pawns on the board can be moved. If pawns were to be places on another pawn of the player,
        # the option is not added to the list
        for pawn_num, loc in enumerate(self.pawn_locations):
            if loc > 0:
                new_location = loc + roll
                if new_location > self.board_size:
                    space_left = new_location % self.board_size
                    if space_left > 4:
                        new_location = -(space_left - 2)
                    else:
                        new_location = -space_left
                if new_location not in self.pawn_locations:
                    possibilities.append(pawn_num)
        return possibilities

    def get_pawn_locations(self, relative: bool = False, rel_player_num: int = 0) -> list:
        """ Returns a list with the indexes of the player locations. If relative is set to True, all locations are
            relative to the selected player (default is player 0).

        :param relative: If set to False, all location are relative to the player of the object. If set to True, its
        location is relative to the location of the player specified with rel_player_num.
        :param rel_player_num: Player to whom the location are relative to. Not used if relative is set to False
        :return: A list containing the locations of the pawns of the player. Location 0 means the pawns are not yet in
        play. Negative values means the pawns are at the end spaces.
        """
        if relative:
            loc_modifier = self.player_num - rel_player_num
            rel_pawn_locations = []
            for loc in self.pawn_locations:
                if loc <= 0:
                    rel_pawn_locations.append(loc)
                else:
                    loc = loc + (loc_modifier * self.places_between_starts)
                    if loc > self.board_size:
                        rel_pawn_locations.append(loc - self.board_size)
                    elif loc < 0:
                        rel_pawn_locations.append(self.board_size + loc)
                    else:
                        rel_pawn_locations.append(loc)
            return rel_pawn_locations
        else:
            return self.pawn_locations

    def do_move(self, roll, pawn_number):
        """ Moves the specified pawn with the number of spaces specified in roll. If the move is not legal, such as when
            a pawn is brought into play without a 6 or when a player places 2 of their own pawns on each other, an
            Exception is thrown.

        :param roll: Amount of spaces the selected pawn moves.
        :param pawn_number: Index of the pawn to move.
        :return: raises IllegalMoveError if the move suggested is not allowed.
        """

        prev_location = self.pawn_locations
        # Section to bring a pawn into play
        if self.pawn_locations[pawn_number] == 0:
            if roll != self.highest_roll:
                raise IllegalMoveError(pawn_number, self.pawn_locations[pawn_number], roll)
            self.pawn_locations[pawn_number] = 1
        else:
            # Section to move pawns in the end spaces
            if self.pawn_locations[pawn_number] < 0:
                self.pawn_locations[pawn_number] -= roll
                if self.pawn_locations[pawn_number] < - 4:
                    self.pawn_locations[pawn_number] = -4 + (-self.pawn_locations[pawn_number] - 4)
                    if self.pawn_locations[pawn_number] >= 0:
                        self.pawn_locations[pawn_number] = (self.board_size - self.pawn_locations[pawn_number])
            # Section to move pawns that are on the board
            else:
                self.pawn_locations[pawn_number] += roll
                # Section to process pawns entering the end section of the board
                if self.pawn_locations[pawn_number] > self.board_size:
                    space_left = self.pawn_locations[pawn_number] % self.board_size
                    if space_left > 4:
                        # Pawns re-entering the board
                        self.pawn_locations[pawn_number] = -(space_left - 2)
                    else:
                        # Pawns staying on the end sectoin
                        self.pawn_locations[pawn_number] = -space_left

        # This section checks for overlapping pawns
        tmp_set = list(self.pawn_locations)
        tmp_set = list(filter((0).__ne__, tmp_set))
        if len(set(tmp_set)) < len(tmp_set):
            self.pawn_locations = prev_location
            raise IllegalMoveError(pawn_number, self.pawn_locations[pawn_number], roll,
                                   message="Move not legal. Moving this pawn causes the locations of 2 pawns to overlap")

    def get_hit(self, pawn_number):
        """ Selected pawn gets hit and is sent back to the start.

        :param pawn_number: Index of the pawn to be hit
        :return: Hele hoop niets
        """
        self.pawn_locations[pawn_number] = 0


class IllegalMoveError(Exception):
    """ Exception thrown when moves are illegal.

    """
    def __init__(self, pawn_num: int, location: int, roll: int,
                 message: str = "Move not legal. Pawns at location -1 require the highest possible roll to enter the "
                                "board"):
        """ Exception thrown when illegal moves are performed

        :param pawn_num: Pawn number that does the illegal move
        :param location: location of the pawn that does an illegal move
        :param roll: The roll used in the illegal move
        :param message: The message of the illegal move, used for further clarification of the offence
        """
        self.pawn_num = pawn_num
        self.pawn_location = location
        self.message = message
        self.roll = roll
        super(IllegalMoveError, self).__init__(self.message)

    def __str__(self):
        return f'{self.message}\nPawn num {self.pawn_num} at location {self.pawn_location} cant move {self.roll}'


class TestPlayer(unittest.TestCase):
    def test_player_move(self):
        p = Player(0)
        p.do_move(6, 0)
        self.assertEqual(p.get_pawn_locations(), [1, 0, 0, 0], "Expected pawn to have entered board")

    def test_player_illegalmove(self):
        p = Player(0)
        p.do_move(6, 0)
        with self.assertRaises(IllegalMoveError):
            p.do_move(6, 1)


if __name__ == '__main__':
    unittest.main()
