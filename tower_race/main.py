"""
Name - Brendan McMullen
"""

import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import HttpResponse


@ensure_csrf_cookie
def hello_world(request):
    return HttpResponse("Hello World!")

# Define the model architecture
class GoodnessNet(nn.Module):
    def __init__(self, input_size=10, hidden1=64, hidden2=32, output_size=1):
        super(GoodnessNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Remove sigmoid for unbounded regression output
        return x
    
class Game:

    # Load the trained model
    loaded_model = GoodnessNet()
    loaded_model.load_state_dict(torch.load("tower_race/AI-models/goodness_model2.pth"))
    loaded_model.eval()  # Set the model to evaluation mode

    # Test the model on specific examples
    def evaluate_tower(self, tower, model):
        model.eval()
        with torch.no_grad():
            tower_tensor = torch.tensor(tower, dtype=torch.float32).unsqueeze(0) / 100.0  # Normalize
            goodness_score = model(tower_tensor).squeeze().item()
        return goodness_score


    def setup_bricks(self):
        """sets of main and discord pile for start of game"""
        self.discard = []
        self.main_pile = []
        #create main pile of bricks
        for i in range(1, 61):
            self.main_pile.append(i)

    def shuffle_bricks(self, bricks):
        """shuffles a pile"""
        random.shuffle(bricks)

    def check_bricks(self, main_pile, discard):
        """checks if there are any brick remaining in the pile"""
        # if main pile is empty transfer bricks to main pile
        if main_pile == []:
            for i in range(0, (len(discard))):
                main_pile.append(discard[i])
            #clear discard pile
            discard.clear()
            #shuffle main pile
            self.shuffle_bricks(main_pile)
            #transfer top brick to discard pile
            top_brick = self.get_top_brick(main_pile)
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

    def deal_initial_bricks(self, main_pile):
        "deals 10 bricks to player and computer"
        computer_hand = []
        player_hand = []
        for i in range(0, 10):
            #deal brick to computer
            computer_hand.insert(0, main_pile[0])
            #remove brick from pile
            main_pile.pop(0)
            #deal brick to player
            player_hand.insert(0, main_pile[0])
            #remove brick from pile
            main_pile.pop(0)
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

    # def computer_play(tower,main_pile,discard):
    #     """manages computer's strategy"""
    #     print("-----COMPUTER'S TURN-----")
    #     outlier_values = []
    #     #the computer will first look at the brick from the discard pile and attempt to place it

    #     #if the brick is greater than the max of
    #     #the tower -3 the computer puts the new brick at the bottom of the tower
    #     if(discard[0] > (max(tower)) - 3):
    #         print("The computer took", discard[0], "from the dicard pile and replaced a brick.")
    #         find_and_replace(get_top_brick(discard), tower[9], tower, discard)
    #         return

    #     #if the brick is less than the min of
    #     #the tower +3 the computer puts the new brick at the top of the tower
    #     elif (discard[0] < (min(tower)) + 3):
    #         print("The computer took", discard[0], "from the dicard pile and replaced a brick.")
    #         find_and_replace(get_top_brick(discard), tower[0], tower, discard)
    #         return

    #     #if the brick has not yet been placed the computer will search for place where the brick
    #     #can fit between a smaller and larger brick in the correct order
    #     for i in range(2, 10):
    #         if((tower[(i - 2)] < discard[0] and tower[i] > discard[0]) and (tower[(i - 1)] < tower[(i - 2)] or tower[(i - 1)] > tower[i])):
    #             print("The computer took", discard[0], "from the dicard pile and replaced a brick.")
    #             find_and_replace(get_top_brick(discard), tower[(i - 1)], tower, discard)
    #             return
    #     #if the brick has not yet been placed the computer will search for a place in middle of the tower
    #     #where the brick can fit between a smaller and larger brick brick even though the bricks will not be in the correct order
    #     #the computer will only do this is the brick is from 20 and 40.
    #     for i in range(3, 9):
    #         if ((discard[0] in range(20, 41)) and tower[(i - 2)] > discard[0] and tower[i] < discard[0]):
    #             if (tower[(i - 1)] < tower[(i - 2)]) or tower[(i - 1)] > tower[i]:
    #                 print("The computer took", discard[0], "from the dicard pile and replaced a brick.")
    #                 find_and_replace(get_top_brick(discard), tower[(i - 1)], tower, discard)
    #                 return

    #     #if brick could not be placed the computer takes the brick from the main pile
    #     #if the brick is greater than the max of
    #     #the tower the computer puts the new brick at the bottom of the tower
    #     if (main_pile[0] > (max(tower))):
    #         print("The computer took a brick from the main pile and replaced a brick.")
    #         find_and_replace(get_top_brick(main_pile), tower[9], tower, discard)
    #         return

    #     #if the brick is ;ess than the min of
    #     #the tower the computer puts the new brick at the bottom of the tower
    #     elif (main_pile[0] < (min(tower))):
    #         print("The computer took a brick from the main pile and replaced a brick.")
    #         find_and_replace(get_top_brick(main_pile), tower[0], tower, discard)
    #         return

    #     #if the brick has not yet been placed the computer will again search for a place where the brick
    #     #can fit between a smaller and larger brick in the correct order
    #     for i in range(2, 10):
    #         if ((tower[(i - 2)] < main_pile[0] and tower[i] > main_pile[0]) and (
    #                 tower[(i - 1)] < tower[(i - 2)] or tower[(i - 1)] > tower[i])):
    #             print("The computer took a brick from the main pile and replaced a brick.")
    #             find_and_replace(get_top_brick(main_pile), tower[(i - 1)], tower, discard)
    #             return

    #         #if the brick has not yet been placed the computer will again search for a place in middle of the tower
    #         #where the brick can fit between a smaller and larger brick brick even though the bricks will not be in the correct order
    #         #again the computer will only do this is the brick is from 20 and 40.
    #     for i in range(3, 9):
    #         if ((main_pile[0] in range(20, 41)) and tower[(i - 2)] > main_pile[0] and tower[i] < main_pile[0]):
    #             if (tower[(i - 1)] < tower[(i - 2)]) or tower[(i - 1)] > tower[i]:
    #                 print("The computer took a brick from the main pile and replaced a brick.")
    #                 find_and_replace(get_top_brick(main_pile), tower[(i - 1)], tower, discard)
    #                 return
    #     #if the computer could not find an appropriate place for the brick it will attempt to find which
    #     #brick is the greatest outlier in the tower by making a list of the values of the brick
    #     #value minus the value of brick below and that value added to the brick above minus the current brick
    #     #and replace the brick with whatever brick was drawn from the main pile
    #     for i in range(2, 10):
    #         outlier_values.append((tower[(i - 1)] - tower[i]) + (tower[(i - 2)] - tower[i - 1]))
    #     print("The computer took a brick from the main pile and replaced a brick.")
    #     find_and_replace(get_top_brick(main_pile), tower[(outlier_values.index(max(outlier_values)) + 1)], tower, discard)
    #     return


    def calculate_best_replacement(self, test_tower, newbrick, threshold, pile):
        if (newbrick < min(test_tower) and newbrick < 5) or (newbrick < 6 and newbrick < test_tower[1] and newbrick < test_tower[0]):
            best_to_replace = test_tower[0]
            return best_to_replace
        if (newbrick > max(test_tower) and newbrick > 55) or (newbrick > 54 and newbrick > test_tower[8] and newbrick > test_tower[9]):
            best_to_replace = test_tower[9]
            return best_to_replace
        slot = None

        difference = 100
        for i in range(1, len(test_tower) - 1):
                current_brick = test_tower[i]
                difference = test_tower[i+1] - test_tower[i-1]
                if newbrick < test_tower[i+1] and newbrick > test_tower[i-1] and difference > 1 and difference < 20:
                    slot = i
                    if abs((i + 1) * 6 - newbrick) < 10:
                        if current_brick > test_tower[i+1] or current_brick < test_tower[i-1]:
                            best_to_replace = current_brick
                            return best_to_replace
                        # elif i < 5 and newbrick < current_brick:
                        #     best_to_replace = current_brick
                        #     print("brick fits between " + str(test_tower[i-1])+ " and " + str(test_tower[i+1]))
                        #     return best_to_replace
                        # elif newbrick > current_brick:
                        #     best_to_replace = current_brick
                        #     print("brick fits between " + str(test_tower[i-1])+ " and " + str(test_tower[i+1]))
                        #     return best_to_replace
                    
        best_to_replace = None
        score = 0
        highest_score = 0

        before_score = self.evaluate_tower(test_tower, self.loaded_model)
        if best_to_replace == None:
            for i in range(0, len(test_tower)):
                current_brick = test_tower[i]
                test_tower[i] = newbrick
                if test_tower == sorted(test_tower):
                    best_to_replace = current_brick
                    return best_to_replace
                score = self.evaluate_tower(test_tower, self.loaded_model)
                offset = abs((i + 1) * 6 - newbrick)
                if offset > 5:
                    score -= ((offset - 5) * .1)
                if i > 0 and i < len(test_tower) - 1:
                    if current_brick < test_tower[i+1] and current_brick > test_tower[i-1]:
                        score -= .2
                if score > highest_score:
                        highest_score = score
                        best_to_replace = current_brick
                test_tower[i] = current_brick 
            if highest_score > before_score + threshold:
                return best_to_replace
                
            elif (newbrick < min(test_tower)):
                best_to_replace = test_tower[0]
                return best_to_replace
            elif (newbrick > max(test_tower)):
                best_to_replace = test_tower[9]
                return best_to_replace
            
            elif slot != None and (pile == 'main' or difference < 10):
                start = 0 if slot <= 4 else slot - 4
                if test_tower[start:slot] == sorted(test_tower[start:slot]) and newbrick < test_tower[slot]:
                    best_to_replace = test_tower[slot]
                    return best_to_replace
                end = 10 if slot >= 6 else slot + 4
                if test_tower[slot:end] == sorted(test_tower[slot:end]) and newbrick > test_tower[slot]:
                    best_to_replace = test_tower[slot]
                    return best_to_replace
            else:
                return None


    def computer_play(self):
        """manages computer's strategy"""
        print("\n")
        print("-----COMPUTER'S TURN-----")
        test_tower = list(self.computer_tower)

        print('self.discard', self.discard)
        take_from_pile = self.discard
        newbrick = take_from_pile[0]
        best_to_replace = self.calculate_best_replacement(test_tower, newbrick, 0, 'discard')
        if best_to_replace != None:
            take_from_pile.remove(newbrick)
            self.find_and_replace(newbrick, best_to_replace, self.computer_tower)
            return f"The computer took {newbrick} from the discard stack."

        take_from_pile = self.main_pile
        newbrick = take_from_pile[0]

        best_to_replace = self.calculate_best_replacement(test_tower, newbrick, -.15, 'main')
        if best_to_replace != None:
            take_from_pile.remove(newbrick)
            self.find_and_replace(newbrick, best_to_replace, self.computer_tower)
            return f"The computer took {newbrick} from the hidden stack."
            
        return "The computer passes"

        


    def players_turn(self):
        """manages players turns"""
        print("\n")
        print("-----YOUR TURN-----")
        print("Your tower:" , self.player_tower)
        #in stage 1 ask the player if they want a brick from the discard or main pile
        return f"Take {str(self.discard[0])} from the discard pile?\r\nClick 'None' to get a new brick from the main pile."
    

    def take_from_pile(self, pile_type, brick_to_replace):
        #take brick from discard pile
        if pile_type == "discard":
            take_pile = self.discard
        elif pile_type == "main":
            take_pile = self.main_pile
        newbrick = take_pile[0]
        #skip stage 2 and go straight to stage 3
                #make sure this input is an integer

        #replace the correct brick, if the brick is not in the player's tower repeat stage 3
        self.find_and_replace(newbrick, brick_to_replace, self.player_tower)
    
        #end players turn
        self.add_brick_to_discard(brick_to_replace)
        take_pile.remove(newbrick)
        return self.player_tower



        # #tell the player which brick was replaced and remove the brick from the pile
        # print("You replaced", brick_to_replace, "with", newbrick)
        # return
        # stage = 3
        #     elif(player_input == "m"):
        #         #tell the player the hidden brick value
        #         print("you drew", self.main_pile[0], "from the main pile.")
        #         #go to stage 2
        #         stage = 2
        #         break
        #     else:
        #         #if the player give the wrong input repeat stage 1
        #         print("please try again")
        #         continue

        # #in stage 2 ask if the player wants the brick from the main pile
        # while (stage == 2):
        #     take_this_brick = (input(str("Type 'Y' to take this brick\r\nType 'P' to pass")).lower())
        #     take_from_pile = main_pile
        #     #if place takes the brick proceed to stage 3
        #     #take brick from main pile
        #     newbrick = take_from_pile[0]
        #     if (take_this_brick == "y"):
                
        #         #go to stage 3
        #         stage = 3
        #         break
        #     #if the player does not want the brick the turn is skipped
        #     elif(take_this_brick == "p"):
        #         print("You discarded the brick.")
        #         self.add_brick_to_discard(self.get_top_brick(take_from_pile))
        #         return
        #     else:
        #         #if the player gives the wrong input repeat stage 2
        #         print("try again")
        #         continue
        # #in stage 3ask the player what brick they want to replace
        # while(stage == 3):

        #     #make sure this input is an integer
        #     try:
        #         brick_to_replace = int(input("What brick do you want to replace?"))
        #     except ValueError:
        #         #if the input is incorrect repeat stage 3
        #         print("try again")
        #         continue
        #     #replace the correct brick, if the brick is not in the player's tower repeat stage 3
        #     correctinput = self.find_and_replace(newbrick, brick_to_replace, tower, discard)
        #     if correctinput == False:
        #         print(brick_to_replace, "is not in you tower.")
                
        #         continue
        #     else:
        #         #end players turn
        #         self.add_brick_to_discard(brick_to_replace)
        #         take_from_pile.remove(newbrick)
        #         stage = 0
        #         break

        # #tell the player which brick was replaced and remove the brick from the pile
        # print("You replaced", brick_to_replace, "with", newbrick)
        # return


    def top_of_pile(self, pile):
        if pile == 'discard':
            return self.discard[0]
        if pile == 'main':
            return self.main_pile[0]


    def start(self):
        """manages general game play"""
        #set up the main and discard piles
        self.setup_bricks()
        #shuffle the main pile
        self.shuffle_bricks(self.main_pile)
        #transfer top brick from main pile to discard pile
        topbrick = self.get_top_brick(self.main_pile)
        self.add_brick_to_discard(topbrick)
        #deal the computer and player towers
        computer_and_player_towers = self.deal_initial_bricks(self.main_pile)
        #split the d into 2 lists
        self.computer_tower = computer_and_player_towers[0]
        self.player_tower = computer_and_player_towers[1]

        self.Computer_wins = False
        self.Player_wins = False

        return { 'computer_tower': self.computer_tower, 'player_tower': self.player_tower }

        
        
        #commence game play
    def play_computers_turn(self):
        #computers turn
        game_text = self.computer_play()
        #check if computer has won
        Computer_wins = self.check_tower_blaster('computer')
        # check if the main pile is empty
        self.check_bricks(self.main_pile, self.discard)
        # self.players_turn(self.player_tower, self.main_pile, self.discard)
        # #check if player has won
        # Player_wins = self.check_tower_blaster(self.player_tower)
        # #check if the main pile is empty
        self.check_bricks(self.main_pile, self.discard)

        return { 'computer_tower':self.computer_tower, 'game_text': game_text, 'computer_wins': Computer_wins}

        #If the player wins
        if(Player_wins == True):
            print("computer's tower:", self.computer_tower)
            print("your tower:", self.player_tower)
            print("YOU WIN!!")

        # If the computer wins
        if(Computer_wins == True):
            print("computer's tower:", self.computer_tower)
            print("your tower:", self.player_tower)
            print("YOU LOSE!!")


def main():
    game = Game()
    game.start()

if __name__ == '__main__':
    main()
