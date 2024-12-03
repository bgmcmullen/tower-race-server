import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the AI model architecture
class GoodnessNet(nn.Module):
  def __init__(self, input_size=10, hidden1=64, hidden2=32, output_size=1):
    super(GoodnessNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden1)
    self.fc2 = nn.Linear(hidden1, hidden2)
    self.fc3 = nn.Linear(hidden2, output_size)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
    
class AILogic:
  def __init__(self):
    # Load the trained model
    self.loaded_model = GoodnessNet()
    self.loaded_model.load_state_dict(torch.load("tower_race/AI-models/goodness_model2.pth"))

  def evaluate_tower(self, tower, model):
    model.eval()
    with torch.no_grad():
      tower_tensor = torch.tensor(tower, dtype=torch.float32).unsqueeze(0) / 100.0  # Normalize
      goodness_score = model(tower_tensor).squeeze().item()
    return goodness_score
  def calculate_best_replacement(self, test_tower, newbrick, threshold, pile):
    def brick_fits_at_bottom():
      buttom_is_greater_than_twelve = test_tower[0] >= 12 
      first_half_of_tower_is_not_sorted =  test_tower[:5] != sorted(test_tower[:5]) 
      new_brick_min_or_close_to_min = newbrick < min(test_tower) or (newbrick < 6 and newbrick < test_tower[1] and newbrick < test_tower[0])
      return buttom_is_greater_than_twelve and first_half_of_tower_is_not_sorted and new_brick_min_or_close_to_min
    
    def brick_fits_at_top():
      top_is_les_than_48 = test_tower[9] <= 48 
      second_half_of_tower_is_not_sorted =  test_tower[5:] != sorted(test_tower[5:])
      new_brick_max_or_close_to_max = newbrick > max(test_tower) or (newbrick > 54 and newbrick > test_tower[8] and newbrick > test_tower[9])
      return top_is_les_than_48 and second_half_of_tower_is_not_sorted and new_brick_max_or_close_to_max

    # Check if brick losely fits at bottom
    if brick_fits_at_bottom():
      best_to_replace = test_tower[0]
      return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits at the bottom.")

    # Check if brick losely fits at top
    if brick_fits_at_top():
      best_to_replace = test_tower[9]
      return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits at the top.")
    
    # Define variable to store the best brick to replace
    best_to_replace = None

    # Define variable to store the difference between brick which the new brick can fit between
    difference = 100

    # Define variable to store the index the new brick can fit between two bricks
    slot = None

    # Check if new brick can fit between to bricks in tower
    for i in range(1, len(test_tower) - 1):
      
      # Store current brick
      current_brick = test_tower[i]

      # Check if new brick is less than next brick and greater than previous brick
      if newbrick < test_tower[i+1] and newbrick > test_tower[i-1]:

        # Find difference between bricks on either side of current brick
        difference = test_tower[i+1] - test_tower[i-1]

        # Store index new brick can fit
        slot = i

        # Find offset between new brick and most linear value of index
        offset = abs((i + 1) * 6 - newbrick)

        # Check if offset is less than ten and difference is less than 20 
        if offset < 10 and difference < 20:

          # If current_brick does not fit in selected slot replace with new brick
          if current_brick > test_tower[i+1] or current_brick < test_tower[i-1]:
            best_to_replace = current_brick
            return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits between two bricks.")
          
    # Define variable to store AI model's "goodness" score for each possible new tower
    score = 0

    # Define variable to store AI's highest "goodness" score
    highest_score = 0

    # Check the current tower's score "goodness" score
    before_score = self.evaluate_tower(test_tower, self.loaded_model)

    # Check "goodness" score for each possible new tower
    for i in range(1, len(test_tower) - 1):

      # Store current brick
      current_brick = test_tower[i]

      # Temporarily replace with new brick
      test_tower[i] = newbrick

      # If this tower is winning replace this brick
      if test_tower == sorted(test_tower):
        best_to_replace = current_brick
        return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it wins.")
      
      # Get AI model's score for this possible tower
      score = self.evaluate_tower(test_tower, self.loaded_model)

      # Find offset between new brick and most linear value of index
      offset = abs((i + 1) * 6 - newbrick)

      # If offset is large reduce score proportionally
      if offset > 5:
        score -= ((offset - 5) * .1)

      # If current brick already fits reduce score
      if current_brick < test_tower[i+1] and current_brick > test_tower[i-1]:
        score -= .2

      # Store highest socre and current brick
      if score > highest_score:
        highest_score = score
        best_to_replace = current_brick

      # Reset tower
      test_tower[i] = current_brick 

    # If new tower score is greater than current tower by threshold replace brick
    if highest_score > before_score + threshold:
      return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it the most logical move.")
        
    # Reevaluate if brick can fit between two bricks only if a slot was found in previous step and either the difference between the surrounding bricks is high or the hidden pile is selected
    elif slot != None and (pile == 'hidden' or difference > 18):

      # If index is 3 or less:
      # Check if preceding 3 bricks are sorted and replace brick if it can safely free up space
      start = 0 if slot <= 3 else slot - 3
      if  slot < 6 and test_tower[start:slot] == sorted(test_tower[start:slot]) and newbrick < test_tower[slot]:
        best_to_replace = test_tower[slot]
        return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it frees up space above.")
        
      # If index is 6 or greater:
      # Check if following 3 bricks are sorted and replace brick if it can safely free up space
      end = 10 if slot >= 6 else slot + 3
      if slot > 4 and test_tower[(slot + 1):end] == sorted(test_tower[slot:end]) and newbrick > test_tower[slot]:
        best_to_replace = test_tower[slot]
        return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it frees up space below.")
        
      # For any index:
      # Check if entire preceding tower is sorted and replace brick if it can safely free up space 
      if test_tower[:slot] == sorted(test_tower[:slot]) and newbrick < test_tower[slot]:
        best_to_replace = test_tower[slot]
        return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it condenses the lower tower.")
        
      # For any index:
      # Check if entire following tower is sorted and replace brick if it can safely free up space 
      if test_tower[(slot + 1):] == sorted(test_tower[(slot + 1):]) and newbrick > test_tower[slot]:
        best_to_replace = test_tower[slot]
        return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it condenses the upper tower.")
        
    # Check if brick strictly fits at bottom
    elif (newbrick < min(test_tower)):
      best_to_replace = test_tower[0]
      return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits at the bottom after all.")
    
    # Check if brick strictly fits at top
    elif (newbrick > max(test_tower)):
      best_to_replace = test_tower[9]
      return (best_to_replace, f"The computer took {newbrick} from the {pile} stack and replaced {best_to_replace} because it fits at the top after all.")
    else:

      # Return none if no good move is found.
      return None