import numpy as np
from deck import Suit
from strategy import make_bid, select_card
import logging

class Game:
    def __init__(self, winning_score, first_dealer_position):
        self.winning_score = winning_score
        self.first_dealer_position = first_dealer_position
        self.dealer_position = None
        self.n_hands = 0
        self.bid_data = [] # [bid_round, table_position, flipped_card, cards_str, is_alone, points]
        
    def increment_dealer(self):
        """Increments the dealer by one position
        """
        self.dealer_position = np.mod(self.dealer_position + 1, 4) if self.dealer_position is not None else self.first_dealer_position
        
    def winner(self, teams):
        """Returns the game winner, if there is one
        """
        for team in teams:
            if team.game_score >= self.winning_score:
                return team
        return None

_hand_counter = 0
class Hand:
    def __init__(self, players, dealer_position):
        global _hand_counter
        _hand_counter += 1
        self.hand_id = _hand_counter
        self.dealer = players[dealer_position]
        self.bid_position_order = np.roll(range(4), -(dealer_position+1))
        self.bid_order = [players[i] for i in self.bid_position_order]
        self.flipped_card = None
        self.trumps = None
        self.maker = None
        self.bid_type = None
        self.cards_played = []
        self.num_tricks_played = 0
        self.winner = None

    
    def result(self):
        """Returns the result of the hand
        """
        if self.winner != self.maker.team: # euchre
            return 'euchre'
        elif self.maker.is_alone and self.winner.tricks_taken == 5: # lone march
            return 'lone_march'
        elif self.winner.tricks_taken == 5: # march
            return 'march'
        else: # single
            return 'single'
        
    def winner_points(self):
        """Returns the number of game points to the hand winner
        """
        if self.result() == 'lone_march':
            return 4
        elif self.result() == 'march':
            return 2
        elif self.result() == 'euchre':
            return 2
        else:
            return 1
        
class Trick:
    def __init__(self, hand, players, leader_position):
        self.hand = hand
        self.cards_played = []
        self.winner = None
        self.winning_card = None
        self.suit_led = None
        self.leader_position = leader_position
        init_position_order = np.roll(range(4), -leader_position)
        self.trick_position_order = [p for p in init_position_order if not players[p].is_partner_alone]
        self.trick_order = [players[i] for i in self.trick_position_order]
        self.maker_position = self.trick_order.index(hand.maker)
        
    def num_cards_played(self):
        """Returns the number of cards played in the trick
        """
        return len(self.cards_played)
    
    def play(self):
        """Plays a trick of four cards
        """
        for player in self.trick_order:
            # Select a card
            card_played = select_card(player.strategy, player, self)
            assert(card_played is not None and card_played in player.valid_cards(self.hand.trumps, self.suit_led))
            suit_played = card_played.effective_suit(self.hand.trumps)
            
            # Play card onto pile and remove from player's hand
            self.cards_played.append(card_played)
            self.hand.cards_played.append(card_played)
            player.cards_played.append(card_played)
            player.cards = list(set(player.cards) - set([card_played]))
            
            # Add to known empty suits if they didn't follow suit
            if (self.num_cards_played() > 1) and (suit_played != self.suit_led):
                player.known_empty_suits.append(self.suit_led)
                
            # Evaluate the trick leader
            if self.num_cards_played() == 1:
                self.winner = player
                self.winning_card = card_played
                self.suit_led = suit_played
            elif card_played.value_in_trick(self.hand.trumps, self.suit_led) > self.winning_card.value_in_trick(self.hand.trumps, self.suit_led):
                self.winner = player
                self.winning_card = card_played
                
        self.hand.num_tricks_played += 1