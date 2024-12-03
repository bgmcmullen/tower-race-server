import json
from channels.generic.websocket import WebsocketConsumer

from  tower_race.main import Game

class GameConsumer(WebsocketConsumer):
  def connect(self):
    self.game = Game()
    self.accept()

  def disconnect(self, close_code):
    self.close()

  def receive(self, text_data):
    data = json.loads(text_data)
    print('data', data)
    type = data["type"]
    switch = {
      "start_game": self.start_game,
      'take_from_discard': self.take_from_discard,
      'take_from_hidden': self.take_from_hidden,
      'get_top_of_discard': self.get_top_of_discard,
      'get_top_of_hidden': self.get_top_of_hidden_pile
    }
    try:
      handler = switch[type]
      if handler:
        handler(data['payload'])
    except KeyError:
      print('no task')

  def get_top_of_discard(self, payload):
    brick = self.game.top_of_pile("discard")
    self.send_event(brick, "set_next_brick")

  def get_top_of_hidden_pile(self, payload):
    brick = self.game.top_of_pile("hidden")
    self.send_event(brick, "set_next_brick")

  # take a new brick and replace a brick in the tower
  def take_new_brick(self, brick_to_replace, stack):

    # Get updated player tower
    playertower = self.game.take_from_pile(stack, int(brick_to_replace))

    # Send updated player tower
    self.send_event(playertower, "set_player_tower")

    # Check if computer has won
    if self.game.check_tower_blaster('player') == True:
      self.send_event(None, "player_wins")
      return

    # Play computer turn
    computer_turn_status = self.game.play_computers_turn()

    # Send computer status
    self.set_computer_turn_status(computer_turn_status)

    # Check if computer has won
    if self.game.check_tower_blaster('computer') == True:
      self.send_event(None, "computer_wins")


  def take_from_discard(self, brick_to_replace):
    self.take_new_brick(brick_to_replace, 'discard')
  
  def take_from_hidden(self, brick_to_replace):
    self.take_new_brick(brick_to_replace, 'hidden')



  # Begin new game
  def start_game(self, incoming_payload):

    # Get starting tower status
    tower_status = self.game.start()

    # Send player and computer tower status
    self.send_event(tower_status['computer_tower'], "set_computer_tower")
    self.send_event(tower_status['player_tower'], "set_player_tower")

    # Play computer turn
    computer_turn_status = self.game.play_computers_turn()

    # Send computer status
    self.set_computer_turn_status(computer_turn_status)

    # Send computer winning message if applicable
    if computer_turn_status['computer_wins'] == True:
      self.send_event(None, "computer_wins")


  # Send computer tower status and computer play text
  def set_computer_turn_status(self, computer_turn_status):
    self.send_event(computer_turn_status['game_text'], "set_computer_play_message")

    self.send_event(computer_turn_status['computer_tower'], "set_computer_tower")

  # Send event to client.
  def send_event(self, payload, type):
    self.send(text_data=json.dumps({
      'payload': payload,
      'type': type,
    }))

