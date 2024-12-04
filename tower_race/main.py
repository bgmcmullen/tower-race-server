"""
Name - Brendan McMullen
"""

import random


from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import HttpResponse
from tower_race.computer_logic import AILogic

# Send HTTP response for testing and evaluation
@ensure_csrf_cookie
def hello_world(request):
    return HttpResponse("Hello World!")
    
class Game:
    ai_logic = AILogic()

    def setup_bricks(self):
        """sets of hidden and discord pile for start of game"""
        self.discard = []
        self.hidden_pile = []
        #create main pile of bricks
        for i in range(1, 61):
            self.hidden_pile.append(i)

    def shuffle_bricks(self, bricks):
        """shuffles a pile"""
        random.shuffle(bricks)

    def check_bricks(self, hidden_pile, discard):
        """checks if there are any brick remaining in the pile"""
        # if hidden pile is empty transfer bricks to main pile
        if hidden_pile == []:
            for i in range(0, (len(discard))):
                hidden_pile.append(discard[i])
            #clear discard pile
            discard.clear()
            #shuffle main pile
            self.shuffle_bricks(hidden_pile)
            #transfer top brick to discard pile
            top_brick = self.get_top_brick(hidden_pile)
            self.add_brick_to_discard(top_brick, discard)

    def check_tower_blaster(self, which_tower):
        """checks if a tower is stable"""

        if which_tower == 'player':
            tower = self.player_tower

        if which_tower == 'computer':
            tower = self.computer_tower

        if(tower == sorted(tower)):
            return True
        else:
            return False

    def get_top_brick(self, brick_pile):
        """removes the top brick of a pile and returns that bricks"""
        try:
            top_brick = brick_pile[0]
            #if brick is not in tower (which should not happen) return nothing
        except IndexError:
            return
        #remove brick from pile
        brick_pile.pop(0)
        return top_brick

    def deal_initial_bricks(self, hidden_pile):
        "deals 10 bricks to player and computer"
        computer_hand = []
        player_hand = []
        for i in range(0, 10):
            #deal brick to computer
            computer_hand.insert(0, hidden_pile[0])
            #remove brick from pile
            hidden_pile.pop(0)
            #deal brick to player
            player_hand.insert(0, hidden_pile[0])
            #remove brick from pile
            hidden_pile.pop(0)
            #return towers as tuple
        return (computer_hand, player_hand)

    def add_brick_to_discard(self, brick):
        "adds brick to discard pile"
        self.discard.insert(0, brick)

    def find_and_replace(self, new_brick,brick_to_be_replaced,tower):
        """find a brick to replace and replaces the brick, the removed brick is placed in the discard pile"""
        #check if the the brick to be replaced is in tower
        try:
            brick_index = tower.index(brick_to_be_replaced)
        except ValueError:
            #return false if brick in not in the tower
            return False
        #add new brick
        tower[brick_index] = new_brick
        #discard old brick
        self.add_brick_to_discard(brick_to_be_replaced)
        return True

    def computer_play(self):
        """manages computer's strategy"""
        print("\n")
        print("-----COMPUTER'S TURN-----")

        # Make a copy of the tower
        test_tower = list(self.computer_tower)

        take_from_pile = self.discard
        newbrick = take_from_pile[0]
        best_to_replace = self.ai_logic.calculate_best_replacement(test_tower, newbrick, 0, 'discard')
        if best_to_replace != None:
            take_from_pile.remove(newbrick)
            self.find_and_replace(newbrick, best_to_replace[0], self.computer_tower)
            return best_to_replace[1]

        take_from_pile = self.hidden_pile
        newbrick = take_from_pile[0]

        best_to_replace = self.ai_logic.calculate_best_replacement(test_tower, newbrick, -.15, 'hidden')
        if best_to_replace != None:
            take_from_pile.remove(newbrick)
            self.find_and_replace(newbrick, best_to_replace[0], self.computer_tower)
            return best_to_replace[1]
            
        return "The computer passes"  

    def take_from_pile(self, pile_type, brick_to_replace):


        # Set the pile to take a brick from
        if pile_type == "discard":
            take_pile = self.discard
        elif pile_type == "hidden":
            take_pile = self.hidden_pile
        newbrick = take_pile[0]


        # Replace the correct brick
        self.find_and_replace(newbrick, brick_to_replace, self.player_tower)
    
        # End players turn
        take_pile.remove(newbrick)
        return self.player_tower



    def top_of_pile(self, pile):
        if pile == 'discard':
            return self.discard[0]
        if pile == 'hidden':
            return self.hidden_pile[0]


    def start(self):
        """manages general game play"""
        # Set up the hidden and discard piles
        self.setup_bricks()

        # Shuffle the hidden pile
        self.shuffle_bricks(self.hidden_pile)

        # Transfer top brick from main pile to discard pile
        topbrick = self.get_top_brick(self.hidden_pile)
        self.add_brick_to_discard(topbrick)

        # Seal the computer and player towers
        computer_and_player_towers = self.deal_initial_bricks(self.hidden_pile)

        # Split the d into 2 lists
        self.computer_tower = computer_and_player_towers[0]
        self.player_tower = computer_and_player_towers[1]

        self.Computer_wins = False
        self.Player_wins = False

        return { 'computer_tower': self.computer_tower, 'player_tower': self.player_tower }

        
        # Computers turn
    def play_computers_turn(self):
        
        game_text = self.computer_play()

        # Check if computer has won
        Computer_wins = self.check_tower_blaster('computer')

        # Check if the hidden pile is empty
        self.check_bricks(self.hidden_pile, self.discard)

        return { 'computer_tower':self.computer_tower, 'game_text': game_text, 'computer_wins': Computer_wins}


def main():
    game = Game()
    game.start()

if __name__ == '__main__':
    main()
