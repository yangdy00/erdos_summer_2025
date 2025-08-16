class Team:
    def __init__(self, number, name = None):
        self.number = number
        self.name = 'Team' + str(number) if name is None else name
        self.game_score = 0
        self.reset_trick_count()
        
    def reset_trick_count(self):
        """Resets trick counter to zero. Called at the start of a new hand
        """
        self.tricks_taken = 0
        self.tricks_conceded = 0
        
class Player:
    def __init__(self, position, strategy, name = None):
        self.position = position
        self.name = 'Player' + str(position) if name is None else name
        self.team = None
        self.opp_team = None
        self.partner = None
        self.strategy = strategy
        self.n_bids = 0
        self.n_lone_bids = 0
        self.n_pickup_bids = 0
        self.n_orderup_bids = 0
        self.n_downalone_bids = 0
        self.n_open_bids = 0
        self.n_singles = 0
        self.n_marches = 0
        self.n_lone_marches = 0
        self.n_euchred = 0
        self.bid_score = 0
        self.lone_bid_score = 0
        self.pickup_bid_score = 0
        self.orderup_bid_score = 0
        self.downalone_bid_score = 0
        self.open_bid_score = 0
        self.n_experiments = 0
        self.experiment_score = 0
        self.reset_bids()
        
    def trigger_experiment(self):
        self.n_experiments += 1
        self.experiment_triggered = True
        
    def reset_bids(self):
        """Resets the player's cards. Called at the start of a new hand
        """
        self.cards = []
        self.cards_played = []
        self.known_empty_suits = []
        self.is_dealer = False
        self.is_maker = False
        self.is_partner_maker = False
        self.is_alone = False
        self.is_partner_alone = False
        self.experiment_triggered = False
        
    def valid_cards(self, trumps, suit_led):
        """Returns a list of valid cards given the suit that was led in the trick
        """
        if suit_led is None:
            return self.cards

        cards_in_suit = [card for card in self.cards if card.effective_suit(trumps) == suit_led]

        if len(cards_in_suit) > 0:
            return cards_in_suit
        else:
            return self.cards