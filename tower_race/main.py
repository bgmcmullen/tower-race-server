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

# # Define the AI model architecture
# class GoodnessNet(nn.Module):
#     def __init__(self, input_size=10, hidden1=64, hidden2=32, output_size=1):
#         super(GoodnessNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.fc3 = nn.Linear(hidden2, output_size)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
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



    # def calculate_best_replacement(self, test_tower, newbrick, threshold, pile):
    #     def brick_fits_at_bottom():
    #         buttom_is_greater_than_twelve = test_tower[0] >= 12 
    #         first_half_of_tower_is_not_sorted =  test_tower[:5] != sorted(test_tower[:5]) 
    #         new_brick_min_or_close_to_min = newbrick < min(test_tower) or (newbrick < 6 and newbrick < test_tower[1] and newbrick < test_tower[0])
    #         return buttom_is_greater_than_twelve and first_half_of_tower_is_not_sorted and new_brick_min_or_close_to_min
        
    #     def brick_fits_at_top():
    #         top_is_les_than_48 = test_tower[9] <= 48 
    #         second_half_of_tower_is_not_sorted =  test_tower[5:] != sorted(test_tower[5:])
    #         new_brick_max_or_close_to_max = newbrick > max(test_tower) or (newbrick > 54 and newbrick > test_tower[8] and newbrick > test_tower[9])
    #         return top_is_les_than_48 and second_half_of_tower_is_not_sorted and new_brick_max_or_close_to_max

    #     # Check if brick losely fits at bottom
    #     if brick_fits_at_bottom():
    #         best_to_replace = test_tower[0]
    #         return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits at the bottom.")

    #     # Check if brick losely fits at top
    #     if brick_fits_at_top():
    #         best_to_replace = test_tower[9]
    #         return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits at the top.")
        
    #     # Define variable to store the best brick to replace
    #     best_to_replace = None

    #     # Define variable to store the difference between brick which the new brick can fit between
    #     difference = 100

    #     # Define variable to store the index the new brick can fit between two bricks
    #     slot = None

    #     # Check if new brick can fit between to bricks in tower
    #     for i in range(1, len(test_tower) - 1):
                
    #             # Store current brick
    #             current_brick = test_tower[i]

                

    #             # Check if new brick is less than next brick and greater than previous brick
    #             if newbrick < test_tower[i+1] and newbrick > test_tower[i-1]:

    #                 # Find difference between bricks on either side of current brick
    #                 difference = test_tower[i+1] - test_tower[i-1]

    #                 # Store index new brick can fit
    #                 slot = i

    #                 # Find offset between new brick and most linear value of index
    #                 offset = abs((i + 1) * 6 - newbrick)

    #                 # Check if offset is less than ten and difference is less than 20 
    #                 if offset < 10 and difference < 20:

    #                     # If current_brick does not fit in selected slot replace with new brick
    #                     if current_brick > test_tower[i+1] or current_brick < test_tower[i-1]:
    #                         best_to_replace = current_brick
    #                         return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits between two bricks.")
                    
    #     # Define variable to store AI model's "goodness" score for each possible new tower
    #     score = 0

    #     # Define variable to store AI's highest "goodness" score
    #     highest_score = 0

    #     # Check the current tower's score "goodness" score
    #     before_score = self.evaluate_tower(test_tower, self.loaded_model)

    #     # Check "goodness" score for each possible new tower
    #     for i in range(1, len(test_tower) - 1):

    #         # Store current brick
    #         current_brick = test_tower[i]

    #         # Temporarily replace with new brick
    #         test_tower[i] = newbrick

    #         # If this tower is winning replace this brick
    #         if test_tower == sorted(test_tower):
    #             best_to_replace = current_brick
    #             return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it wins.")
            
    #         # Get AI model's score for this possible tower
    #         score = self.evaluate_tower(test_tower, self.loaded_model)

    #         # Find offset between new brick and most linear value of index
    #         offset = abs((i + 1) * 6 - newbrick)

    #         # If offset is large reduce score proportionally
    #         if offset > 5:
    #             score -= ((offset - 5) * .1)
   
    #         # If current brick already fits reduce score
    #         if current_brick < test_tower[i+1] and current_brick > test_tower[i-1]:
    #             score -= .2

    #         # Store highest socre and current brick
    #         if score > highest_score:
    #             highest_score = score
    #             best_to_replace = current_brick

    #         # Reset tower
    #         test_tower[i] = current_brick 

    #     # If new tower score is greater than current tower by threshold replace brick
    #     if highest_score > before_score + threshold:
    #         return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it the most logical move.")
            
    #     # Reevaluate if brick can fit between two bricks only if a slot was found in previous step and either the difference between the surrounding bricks is high or the hidden pile is selected
    #     elif slot != None and (pile == 'hidden' or difference > 18):

    #         # If index is 3 or less:
    #         # Check if preceding 3 bricks are sorted and replace brick if it can safely free up space
    #         start = 0 if slot <= 3 else slot - 3
    #         if  slot < 6 and test_tower[start:slot] == sorted(test_tower[start:slot]) and newbrick < test_tower[slot]:
    #             best_to_replace = test_tower[slot]
    #             return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it frees up space above.")
            
    #         # If index is 6 or greater:
    #         # Check if following 3 bricks are sorted and replace brick if it can safely free up space
    #         end = 10 if slot >= 6 else slot + 3
    #         if slot > 4 and test_tower[(slot + 1):end] == sorted(test_tower[slot:end]) and newbrick > test_tower[slot]:
    #             best_to_replace = test_tower[slot]
    #             return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it frees up space below.")
            
    #         # For any index:
    #         # Check if entire preceding tower is sorted and replace brick if it can safely free up space 
    #         if test_tower[:slot] == sorted(test_tower[:slot]) and newbrick < test_tower[slot]:
    #             best_to_replace = test_tower[slot]
    #             return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it condenses the lower tower.")
            
    #         # For any index:
    #         # Check if entire following tower is sorted and replace brick if it can safely free up space 
    #         if test_tower[(slot + 1):] == sorted(test_tower[(slot + 1):]) and newbrick > test_tower[slot]:
    #             best_to_replace = test_tower[slot]
    #             return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it condenses the upper tower.")
            
    #     # Check if brick strictly fits at bottom
    #     elif (newbrick < min(test_tower)):
    #             best_to_replace = test_tower[0]
    #             return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits at the bottom after all.")
        
    #     # Check if brick strictly fits at top
    #     elif (newbrick > max(test_tower)):
    #             best_to_replace = test_tower[9]
    #             return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits at the top after all.")
    #     else:

    #         # Return none if no good move is found.
    #         return None


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
