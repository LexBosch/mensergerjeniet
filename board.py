from player import Player

class Board:
    def __init__(self, num_players: int = 4, places_between_starts: int = 10):
        self.players = []
        self.num_players = num_players
        self.space = places_between_starts
        for i in range(num_players):
            self.players.append(Player(i, num_players=num_players))

    def get_all_location(self) -> list:
        return [self.players[i].get_pawn_locations() for i in range(self.num_players)]

    def do_player_move(self, player_num: int, pawn_number: int, roll: int):
        self.players[player_num].do_move(pawn_number=pawn_number, roll=roll)


    def show_board(self):
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


def main():
    game = Board()
    game.do_player_move(player_num=0, pawn_number=0, roll=6)
    game.do_player_move(player_num=0, pawn_number=0, roll=6)
    game.do_player_move(player_num=3, pawn_number=0, roll=6)

    print(game.show_board())

if __name__ == '__main__':
    main()