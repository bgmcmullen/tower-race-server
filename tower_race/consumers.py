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
      "test_message": self.test,
      "start_game": self.start_game,
      'take_from_discard': self.take_from_discard,
      'take_from_main': self.take_from_main,
      'get_top_of_discard': self.get_top_of_discard,
      'get_top_of_main': self.get_top_of_main_pile
    }
    try:
      handler = switch[type]
      if handler:
        handler(data['payload'])
    except KeyError:
      print('no task')


  def get_top_of_discard(self, payload):
    brick = self.game.top_of_pile("discard")
    self.send(text_data=json.dumps({
      'payload': brick,
      'type': "set_next_brick",
    }))

  def get_top_of_main_pile(self, payload):
    brick = self.game.top_of_pile("main")
    self.send(text_data=json.dumps({
      'payload': brick,
      'type': "set_next_brick",
    }))


  def take_from_discard(self, payload):
    playertower = self.game.take_from_pile('discard', int(payload))
    self.send(text_data=json.dumps({
      'payload': playertower,
      'type': "set_player_tower",
    }))

    if self.game.check_tower_blaster('player') == True:
      self.send(text_data=json.dumps({
        'payload': None,
        'type': "player_wins",
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

    if self.game.check_tower_blaster('computer') == True:
      self.send(text_data=json.dumps({
        'payload': None,
        'type': "computer_wins",
      }))
  
  def take_from_main(self, payload):
    playertower = self.game.take_from_pile('main', int(payload))
    self.send(text_data=json.dumps({
      'payload': playertower,
      'type': "set_player_tower",
    }))

    if self.game.check_tower_blaster('player') == True:
      self.send(text_data=json.dumps({
        'payload': None,
        'type': "player_wins",
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

    if self.game.check_tower_blaster('computer') == True:
      self.send(text_data=json.dumps({
        'payload': None,
        'type': "computer_wins",
      }))



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
