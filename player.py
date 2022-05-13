end_spaces_fast = {1:-2, 2:-3, 3:-4,4:-5,5:-4,6:-3}

class Player:
    def __init__(self, player_num: int, num_players: int = 4, places_between_starts: int = 10):
        self.amount_players = num_players
        self.places_between_starts = places_between_starts
        self.pawn_locations = [-1, -1, -1, -1]
        self.board_size = (num_players * places_between_starts)
        self.player_num = player_num
        self.end_space = self.board_size - 1 - (player_num*10)
    # def get_possible_moves(self, roll):

    def get_pawn_locations(self):
        return self.pawn_locations

    def do_move(self, roll, pawn_number):

        if self.pawn_locations[pawn_number] == -1:
            if roll != 6:
                raise IllegalMoveError(pawn_number, self.pawn_locations[pawn_number], roll)
            self.pawn_locations[pawn_number] = self.player_num*10
        else:
            prev_loc =  self.pawn_locations[pawn_number]
            self.pawn_locations[pawn_number] += roll
            if self.pawn_locations[pawn_number] > self.end_space:
                space_left = (prev_loc + roll) % self.end_space
                self.pawn_locations[pawn_number] = end_spaces_fast[space_left]
    # def get_hit(self, pawn_location):



class IllegalMoveError(Exception):
    def __init__(self, pawn_num:int, location:int, roll:int, message:int ="Move not legal. Pawns at location -1 require a roll of 6 to enter the board"):
        self.pawn_num = pawn_num
        self.pawn_location = location
        self.message = message
        self.roll = roll
        super(IllegalMoveError, self).__init__(self.message)

    def __str__(self):
        return f'{self.message}\nPawn num {self.pawn_num} at location {self.pawn_location} cant move {self.roll}'


def main():
    player1 = Player()
    player1.do_move(1, 1)

if __name__ == '__main__':
    main()
