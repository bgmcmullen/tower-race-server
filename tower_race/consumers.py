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
    type = data["type"]
    switch = {
      "test_message": self.test,
      "start_game": self.start_game
    }
    handler = switch[type]
    if handler:
      handler(data['payload'])


  def start_game(self, incoming_payload):
    payload = self.game.start()
    self.send(text_data=json.dumps({
      'payload': payload['computer_tower'],
      'type': "set_computer_tower",
    }))
    self.send(text_data=json.dumps({
      'payload': payload['player_tower'],
      'type': "set_player_tower",
    }))
    computer_turn_status = self.game.play_computers_turn()
    self.send(text_data=json.dumps({
      'payload': computer_turn_status['game_text'],
      'type': "set_computer_play_message",
    }))

    self.send(text_data=json.dumps({
      'payload': computer_turn_status['computer_tower'],
      'type': "set_computer_tower",
    }))
    if computer_turn_status['computer_wins'] == True:
      self.send(text_data=json.dumps({
        'payload': None,
        'type': "computer_wins",
      }))


  def test(self, payload):
    print('connection test')
    self.send(text_data=json.dumps({
      'payload': 'web socket is working',
      'type': "test",
    }))
