import numpy as np
import random
from deck import Card, Suit
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('grouped_cleaned_copy.csv')

# Define features and targets
features = ['num_trump', 'right_bower', 'left_bower', 'trump_ace', 'trump_king', 
            'trump_queen', 'trump_ten', 'trump_nine', 'offsuit_aces', 'flipped_jack', 
            'flipped_ace', 'seat_position', 'bidding_round']
X = df[features]
y_bid = df['should_bid']
y_alone = df['should_alone']

# Split data (same splits for both targets)
X_train, X_test, _, _ = train_test_split(X, y_bid, test_size=0.2, random_state=42)
_, _, y_bid_train, y_bid_test = train_test_split(X, y_bid, test_size=0.2, random_state=42)
_, _, y_alone_train, y_alone_test = train_test_split(X, y_alone, test_size=0.2, random_state=42)

# Model Pipelines ---------------------------------------------------

# 1. BIDDING MODELS
bid_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

bid_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# 2. GOING ALONE MODELS
alone_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

alone_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Training ----------------------------------------------------------

# Train bidding models
bid_logreg.fit(X_train, y_bid_train)
bid_rf.fit(X_train, y_bid_train)

# Train going-alone models
alone_logreg.fit(X_train, y_alone_train)
alone_rf.fit(X_train, y_alone_train)

# Evaluation --------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    return model

# Evaluate all models
bid_logreg = evaluate_model(bid_logreg, X_test, y_bid_test, "Bidding Logistic Regression")
bid_rf = evaluate_model(bid_rf, X_test, y_bid_test, "Bidding Random Forest")

alone_logreg = evaluate_model(alone_logreg, X_test, y_alone_test, "Going Alone Logistic Regression")
alone_rf = evaluate_model(alone_rf, X_test, y_alone_test, "Going Alone Random Forest")

# Prediction Example ------------------------------------------------

def make_prediction(model, input_data, model_name):
    sample_input = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(sample_input)
    proba = model.predict_proba(sample_input)
    return prediction

def pickup_and_discard(cards, flipped_card):
    """Picks up the flipped card and discards one card according to their strategy
    """
    # Append flipped card to hand
    cards.append(flipped_card)

    # Decide the weakest card
    card_values = [card.value_in_trick(flipped_card.suit, flipped_card.suit) for card in cards]
    weakest_card = cards[np.argmin(card_values)]
    
    # Decide the weakest card if the offsuit is bare
    bare_offsuit_cards = get_bare_offsuits(cards, flipped_card.suit, excluded_ranks = ['A'])
    bare_offsuit_card_values = [card.value_in_trick(flipped_card.suit, flipped_card.suit) for card in bare_offsuit_cards]
    weakest_card_bare_offsuit = bare_offsuit_cards[np.argmin(bare_offsuit_card_values)] if len(bare_offsuit_cards) > 0 else weakest_card

    # Remove it from hand
    cards.remove(weakest_card_bare_offsuit)
    return cards

def get_elig_suits(bid_round, flipped_suit):
    """Returns list of eligible suits a player can bid
    """
    if bid_round == 0:
        return [flipped_suit]
    else:
        return [Suit(s) for s in 'hdcs' if s != flipped_suit.abbrev]
    
def get_action(bid_round, table_position):
    """Returns the action a player can take in a bid round
    """
    if bid_round == 0:
        if table_position in [0, 2]:
            return 'order_up'
        elif table_position == 1:
            return 'down_alone'
        else:
            return 'pick_up'
    else:
        return 'open'

def get_valid_cards(cards, trumps, suit_led):
    """Returns a list of valid cards given the suit that was led in the trick
    """
    if suit_led is None:
        return cards

    cards_in_suit = [card for card in cards if card.effective_suit(trumps) == suit_led]

    if len(cards_in_suit) > 0:
        return cards_in_suit
    else:
        return cards
        
def num_trumps(cards, trumps):
    """Returns the number of trump cards in a set of cards
    """
    return len([card for card in cards if card.effective_suit(trumps) == trumps])

def num_suits(cards, trumps):
    """Returns the number of distinct suits in a set of cards
    """
    return len(set([card.effective_suit(trumps).abbrev for card in cards]))

def num_known_in_suit(suit, known_cards, trumps):
    """Returns the number of known cards in a suit
    """
    return len([card for card in known_cards if card.effective_suit(trumps) == suit])

def num_known_in_hand(cards, cards_played_in_hand, trumps):
    """Returns the number of known cards in the suit of each card in a set of cards 
    """
    known_cards = cards + cards_played_in_hand
    return [num_known_in_suit(card.suit, known_cards, trumps) for card in cards]

def would_lead_trick(cards, trumps, suit_led, winning_card):
    """Returns a Boolean for whether each card in a set of cards would lead the trick
    """
    if winning_card is None:
        return [True] * len(cards)
    else:
        winning_value = winning_card.value_in_trick(trumps, suit_led)
        card_values = [card.value_in_trick(trumps, suit_led) for card in cards]
        return [val > winning_value for val in card_values]

def num_of_suit_in_hand(cards, trumps):
    """Returns a list with the number of each suit in a set of cards
    """
    return [[c.effective_suit(trumps) for c in cards].count(card.effective_suit(trumps)) for card in cards]

def get_bare_offsuits(cards, trumps, excluded_ranks = []):
    """Returns a list of cards in a set of cards that are bare offsuits (ie not trumps, and the only card of that suit in the set)
    """
    suit_counts = num_of_suit_in_hand(cards, trumps)
    return [card for card, count in zip(cards, suit_counts) if (card.effective_suit(trumps) != trumps) and (count == 1) and (card.rank not in excluded_ranks)]

def get_best_trump_left(cards_played, trumps):
    """Returns the best remaining trump based on the cards already played
    """
    left_mapping = {'h': Suit('d'), 'd': Suit('h'), 'c': Suit('s'), 's': Suit('c')}
    left_suit = left_mapping[trumps.abbrev]
    trumps_ordered = [Card('J', trumps), Card('J', left_suit), Card('A', trumps), Card('K', trumps), Card('Q', trumps), Card('T', trumps), Card('9', trumps), Card('8', trumps), Card('7', trumps)]
    trumps_left = [t for t in trumps_ordered if t not in cards_played]
    
    return trumps_left[0] if len(trumps_left) > 0 else None

def opponents_out_of_trumps(player, trick):
    """Returns True if all opponents are out of trumps, otherwise False
    """
    for p in trick.trick_order:
        if p not in [player, player.partner] and (trick.hand.trumps not in p.known_empty_suits):
            return False
    return True

def flip_with_probability(value, prob=0.1):
    return not value if random.random() < prob else value

def make_bid(strategy, cards, bid_round, table_position, flipped_card):
    """Returns player's bid as tuple (bid, alone, suit, this_suit)
    """
    assert((bid_round in [0,1]) and (table_position in [0,1,2,3]))

    # Get possible suits and actions
    elig_suits = get_elig_suits(bid_round, flipped_card.suit)
    action = get_action(bid_round, table_position)

    # Score hand based on bidding each eligible suit
    best_suit = ''
    best_score = -1

    for suit in elig_suits:
        tmp_cards = cards.copy()

        # Mentally pick up the card if you're the dealer
        if action == 'pick_up':
            tmp_cards = pickup_and_discard(tmp_cards, flipped_card)

        # Evaluate a hand score for the purpose of picking the best suit
        method_for_suit_eval = 'complex' if strategy == 'scorecard_complex' else 'simple'
        hand_score = get_hand_score(method_for_suit_eval, tmp_cards, suit, bid_round, table_position, flipped_card)

        if hand_score > best_score:
            best_suit = suit
            best_score = hand_score

    this_suit = best_suit if best_suit else flipped_card.suit
        
    if strategy == 'scorecard_simple':
        
        # Set return values (bid, alone, suit) based on score heuristics
        thresholds = {0: {0: [8,9], # order_up
                          1: [np.nan,8], # down_alone
                          2: [8,11], # order_up
                          3: [6,9]}, # pick_up
                      1: {0: [7,9], # open
                         1: [6,9], # open
                         2: [7,9], # open
                         3: [6,9]} # open
                     }
        if best_score >= thresholds[bid_round][table_position][1]:
            return True, True, best_suit, this_suit
        elif best_score >= thresholds[bid_round][table_position][0]:
            return True, False, best_suit, this_suit
        else:
            return False, False, None, this_suit
            
    elif strategy == 'scorecard_complex':
        
        # Set return values (bid, alone, suit) based on score heuristics
        thresholds = {0: {0: [26,30], # order_up
                  1: [16,25], # down_alone
                  2: [31,34], # order_up
                  3: [16,25]}, # pick_up
              1: {0: [15,26], # open
                 1: [20,28], # open
                 2: [15,26], # open
                 3: [17,28]} # open
             }
        
        if best_score >= thresholds[bid_round][table_position][1]:
            return True, True, best_suit, this_suit
        elif best_score >= thresholds[bid_round][table_position][0]:
            # res = flip_with_probability(True)
            # best_suit = best_suit if res == True else None
            # return res, False, best_suit, this_suit
            return True, False, best_suit, this_suit
        else:
            # res = flip_with_probability(False)
            # best_suit = this_suit if res == True else None
            # return res, False, best_suit, this_suit
            return False, False, None, this_suit
    elif strategy == 'user_input':
        
        # Ask the user
        user_bid = '#'
        user_alone = '#'
        
        if bid_round == 0:
            while user_bid not in 'YN':
                user_bid = input('Bid? (Y/N)')
                suit = flipped_card.suit
        else:
            while user_bid not in 'HDCSN':
                elig_abbrevs = '/'.join([s.abbrev.upper() for s in elig_suits])
                user_bid = input(f'Bid? ({elig_abbrevs}/N)')
                suit = Suit(user_bid.lower())
        
        bid = (user_bid != 'N')
        
        if bid:
            if (bid_round == 0) and (table_position == 1):
                alone = True
            else:
                while user_alone not in 'YN':
                    user_alone = input('Alone? (Y/N)')
                alone = (user_alone == 'Y')
        else:
            alone = False
            suit = None

        return bid, alone, suit, this_suit
    
    elif strategy == 'random':
        bid_prob = 0.2
        alone_given_bid_prob = 0
        this_suit = random.choice(elig_suits)
        return (random.random() < bid_prob, random.random() < bid_prob * alone_given_bid_prob, this_suit, this_suit)

    elif strategy == 'new':
        if bid_round == 0:
            sample_hand = [num_trumps(cards, flipped_card.suit), int(any([card.is_right_bower(flipped_card.suit) for card in cards])), int(any([card.is_left_bower(flipped_card.suit) for card in cards])), int(any([(card.is_trump_suit(flipped_card.suit) and (card.rank == 'A')) for card in cards])), int(any([(card.is_trump_suit(flipped_card.suit) and (card.rank == 'K')) for card in cards])), int(any([(card.is_trump_suit(flipped_card.suit) and (card.rank == 'Q')) for card in cards])), int(any([(card.is_trump_suit(flipped_card.suit) and (card.rank == 'T')) for card in cards])), int(any([(card.is_trump_suit(flipped_card.suit) and (card.rank == '9')) for card in cards])), sum([card.is_offsuit_ace(flipped_card.suit) for card in cards]), int(flipped_card.rank == 'J'), int(flipped_card.rank == 'A'), (table_position + 1) % 4, 0]
            bid  = make_prediction(bid_logreg, sample_hand, "Bidding Logistic Regression")
            alone = make_prediction(alone_logreg, sample_hand, "Going Alone Logistic Regression")
            suit = flipped_card.suit if bid else None
            this_suit = flipped_card.suit
            return bid, (alone and bid), suit, this_suit
        else:
            best_suit = ''
            best_score = -1
            for suit in elig_suits:
                hand_score = get_hand_score('simple', cards, suit, bid_round, table_position, flipped_card)
                if hand_score > best_score:
                    best_suit = suit
                    best_score = hand_score
            sample_hand = [num_trumps(cards, best_suit), int(any([card.is_right_bower(best_suit) for card in cards])), int(any([card.is_left_bower(best_suit) for card in cards])), int(any([(card.is_trump_suit(best_suit) and (card.rank == 'A')) for card in cards])), int(any([(card.is_trump_suit(best_suit) and (card.rank == 'K')) for card in cards])), int(any([(card.is_trump_suit(best_suit) and (card.rank == 'Q')) for card in cards])), int(any([(card.is_trump_suit(best_suit) and (card.rank == 'T')) for card in cards])), int(any([(card.is_trump_suit(best_suit) and (card.rank == '9')) for card in cards])), sum([card.is_offsuit_ace(best_suit) for card in cards]), 0, 0, (table_position + 1) % 4, 1]
            bid  = make_prediction(bid_logreg, sample_hand, "Bidding Logistic Regression")
            alone = make_prediction(alone_logreg, sample_hand, "Going Alone Logistic Regression")
            suit = best_suit if bid else None
            this_suit = best_suit
            return bid, (alone and bid), suit, this_suit
                
            
        
    
    return ValueError('Invalid strategy')

def count_above(cards, trumps, threshold):
    """Returns the number of cards in a list above a given threshold
    """
    return sum([card.value_in_trick(trumps, trumps) >= threshold for card in cards])

def top_offsuit_rank(cards, trumps):
    """Returns the highest-ranking offsuit from a list of cards
    """
    return max([card.value_in_trick(trumps, trumps) for card in cards if card.effective_suit(trumps) != trumps])
  
def get_hand_score(method, cards, trumps, bid_round, table_position, flipped_card):
    """Scores a player's hand based on scoring heuristics
    """
    hand_score = 0
    
    if method == 'simple':
        rules = {'bower_score': 4, 'trump_score': 2, 'offsuit_ace_score': 1}
        
        for card in cards:
            if card.is_bower(trumps):
                hand_score += rules['bower_score']
            elif card.is_trump_suit(trumps):
                hand_score += rules['trump_score']
            elif card.is_offsuit_ace(trumps):
                hand_score += rules['offsuit_ace_score']
            else:
                continue
    else:
        rules = {'n_trumps': {1:1, 2:2, 3:6, 4:9, 5:12},
            'c_value': {216:10, 215:8, 214:6, 213:5, 212:4, 210:4, 209:3, 208:3, 207:3, 14:4, 13:1},
            'flipped_opp': {'J':-5, 'A':-2}}
    
        n_trumps = num_trumps(cards, trumps)
        hand_score += rules['n_trumps'][n_trumps] if n_trumps in rules['n_trumps'] else 0

        for card in cards:
            c_value = card.value_in_trick(trumps, trumps)
            hand_score += rules['c_value'][c_value] if c_value in rules['c_value'] else 0

        if (bid_round == 0) and (table_position in [0, 2]):
            hand_score += rules['flipped_opp'][flipped_card.rank] if flipped_card.rank in rules['flipped_opp'] else 0
    
    return hand_score
   

def pick_card(cards, trumps, suit_led, winning_card, objective, tiebreaker=None, tiebreak_reverse=False, suits_allowed=None):
    """Returns the card to play based on a simple objective. Tiebreaker is highest first unless reversed
    """

    if suits_allowed is None:
        suits_allowed = [Suit('h'), Suit('d'), Suit('c'), Suit('s')]

    ref_suit = suit_led if winning_card is not None else trumps
    card_values = [card.value_in_trick(trumps, ref_suit) if card.effective_suit(trumps) in suits_allowed else np.nan for card in cards]
    winning_card_value = winning_card.value_in_trick(trumps, suit_led) if winning_card is not None else -1

    if tiebreaker is not None:
        tiebreak_sign = -1 if tiebreak_reverse else 1 
        card_values = [c + tiebreak_sign * n / 100 for c, n in zip(card_values, tiebreaker)]

    if objective == 'strongest':
        return cards[np.nanargmax(card_values)]
    elif objective == 'weakest':
        return cards[np.nanargmin(card_values)]
    elif objective == 'weakest_winning':
        winning_values = [val if val > winning_card_value else np.nan for val in card_values]
        return cards[np.nanargmin(winning_values)] if not np.isnan(winning_values).all() else None
    elif objective == 'weakest_trump':
        trump_values = [val if card.effective_suit(trumps) == trumps else np.nan for card, val in zip(cards, card_values)]
        return cards[np.nanargmin(trump_values)] if not np.isnan(trump_values).all() else None
    elif objective == 'strongest_offsuit':
        offsuit_values = [val if card.effective_suit(trumps) != trumps else np.nan for card, val in zip(cards, card_values)]
        return cards[np.nanargmax(offsuit_values)] if not np.isnan(offsuit_values).all() else None
    
    raise ValueError("Invalid value of objectve")
        
def select_card(strategy, player, trick):
    """Selects a card to play
    """

    # Trick status
    valid_cards = player.valid_cards(trick.hand.trumps, trick.suit_led)
    trick_num = trick.hand.num_tricks_played + 1
    card_num = trick.num_cards_played() + 1
    winning_card_value = trick.winning_card.value_in_trick(trick.hand.trumps, trick.suit_led) if trick.winning_card is not None else -1 #TODO: turn into function

    # Position indicators
    is_first_card = (card_num == 1)
    is_middle_card = (card_num == 2) or ((card_num == 3) and not trick.hand.maker.is_alone)
    is_last_card = (card_num == 4) or ((card_num == 3) and trick.hand.maker.is_alone)

    # Bid indicators
    is_partner_winning = (trick.winner == player.partner)
    player_made_bid = (trick.hand.maker == player)
    team_made_bid = (trick.hand.maker.team == player.team)
    partner_made_bid = team_made_bid and not player_made_bid
    opponents_made_bid = not team_made_bid

    # Hand composition variables
    cards_would_lead_trick = would_lead_trick(valid_cards, trick.hand.trumps, trick.suit_led, trick.winning_card)
    num_cards_suit_known = num_known_in_hand(valid_cards, trick.hand.cards_played, trick.hand.trumps)
    num_trumps_in_hand = num_trumps(valid_cards, trick.hand.trumps)
    num_offsuits_in_hand = len(valid_cards) - num_trumps_in_hand
    all_cards_would_win = all(cards_would_lead_trick)
    all_cards_would_lose = not any(cards_would_lead_trick)
    bare_offsuit_cards = get_bare_offsuits(valid_cards, trick.hand.trumps, excluded_ranks = ['A'])
    has_right_bower = any([card.is_right_bower(trick.hand.trumps) for card in valid_cards])
    best_trump_left = get_best_trump_left(trick.hand.cards_played, trick.hand.trumps)
    has_best_trump_left = False if best_trump_left is None else any([card == best_trump_left for card in valid_cards])
    partner_may_have_trumps = (trick.hand.trumps not in player.partner.known_empty_suits)
    are_opponents_out_of_trumps = opponents_out_of_trumps(player, trick)
    only_player_has_trumps = are_opponents_out_of_trumps and num_trumps_in_hand >= 1 and (player.is_alone or not partner_may_have_trumps)

    # Possible cards to select
    strongest_card = pick_card(valid_cards, trick.hand.trumps, trick.suit_led, trick.winning_card, 'strongest')
    weakest_card = pick_card(valid_cards, trick.hand.trumps, trick.suit_led, trick.winning_card, 'weakest')
    weakest_card_bare_offsuit = pick_card(bare_offsuit_cards, trick.hand.trumps, trick.suit_led, trick.winning_card, 'weakest') if (len(bare_offsuit_cards) > 0) and (num_trumps_in_hand > 0) and (trick_num != 4) else weakest_card
    weakest_winning_card = pick_card(valid_cards, trick.hand.trumps, trick.suit_led, trick.winning_card, 'weakest_winning')
    weakest_trump = pick_card(valid_cards, trick.hand.trumps, trick.suit_led, trick.winning_card, 'weakest_trump')
    strongest_offsuit = pick_card(valid_cards, trick.hand.trumps, trick.suit_led, trick.winning_card, 'strongest_offsuit')
    strongest_offsuit_fewest_known = pick_card(valid_cards, trick.hand.trumps, trick.suit_led, trick.winning_card, 'strongest_offsuit', tiebreaker=num_cards_suit_known, tiebreak_reverse=True)
    strongest_offsuit_most_known = pick_card(valid_cards, trick.hand.trumps, trick.suit_led, trick.winning_card, 'strongest_offsuit', tiebreaker=num_cards_suit_known, tiebreak_reverse=False)
    strongest_offsuit_partner_out = pick_card(valid_cards, trick.hand.trumps, trick.suit_led, trick.winning_card, 'strongest_offsuit', suits_allowed=player.partner.known_empty_suits)

    # Possible card indicators
    weakest_winning_card_is_trump = (weakest_winning_card is not None) and (weakest_winning_card.effective_suit(trick.hand.trumps) == trick.hand.trumps)
    weakest_winning_card_is_offsuit = (weakest_winning_card is not None) and (weakest_winning_card.effective_suit(trick.hand.trumps) != trick.hand.trumps)

    if len(valid_cards) == 1:
        # Only one valid card to play
        return valid_cards[0]

    if strategy == 'random':
        return random.choice(valid_cards)
    
    else:
        if is_first_card:
            # Lead trumps if all others are out
            if only_player_has_trumps:
                return strongest_card

            if trick_num in [1,2,3]:
                # Lead strongest if we bid, and I have a strong enough trump hand
                if (partner_made_bid and has_right_bower) or (player_made_bid and (trick_num == 1) and has_right_bower) or (player_made_bid and (num_trumps_in_hand >= 2)) or (num_offsuits_in_hand == 0):
                    return strongest_card
                # Lead weakest trump for partner to overtrump
                elif partner_made_bid and (num_trumps_in_hand >= 1):
                    return weakest_trump
                # Lead strongest offsuit if partner made bid
                # TODO: Don't lead an offsuit if all other players are out, and opponents may have trumps
                elif team_made_bid and partner_may_have_trumps and strongest_offsuit_partner_out is not None:
                    return strongest_offsuit_partner_out
                elif partner_made_bid:
                    return strongest_offsuit_most_known
                # Lead strongest offsuit if opponents made bid
                elif num_offsuits_in_hand >= 1:
                    return strongest_offsuit_partner_out or strongest_offsuit_most_known or weakest_trump
            else: #trick_num == 4
                # Lead strongest card if I have a strong hand
                if (num_trumps_in_hand == 2) or has_best_trump_left:
                    return strongest_card
                # Lead strongest offsuit if partner made bid
                elif num_trumps_in_hand == 0:
                    if team_made_bid and partner_may_have_trumps and strongest_offsuit_partner_out is not None:
                        return strongest_offsuit_partner_out
                    else:
                        return strongest_offsuit_fewest_known
                else: # One trump, one offsuit
                    # Try to win both tricks if we have the point already
                    if player.team.tricks_taken == 3:
                        if team_made_bid and partner_may_have_trumps and strongest_offsuit_partner_out is not None:
                            return strongest_offsuit_partner_out
                        else:
                            return strongest_card
                    elif not partner_may_have_trumps and (player.team.tricks_conceded == 2):
                        return strongest_card
                    # Otherwise lead the weakest card and try to win the last trick
                    else:
                        return weakest_card

        elif is_middle_card:
            # Throw off if all cards would lose
            if all_cards_would_lose: 
                return weakest_card_bare_offsuit
            # Throw off if partner is winning with a strong card
            # TODO: also throw off if partner is leading and player has only one valuable card
            elif is_partner_winning and (winning_card_value >= 213 or 0 <= winning_card_value <= 114):
                return weakest_card_bare_offsuit
            # Throw off if player made bid and has a skinny trumps hand
            elif player_made_bid and weakest_winning_card_is_trump and ( \
                    (trick_num == 1) and (num_trumps_in_hand <= 2) or \
                    (trick_num in [2,3,4] and (num_trumps_in_hand <= 1) and (player.team.tricks_conceded != 2)) \
                ):
                return weakest_card_bare_offsuit
            # Play strongest offsuit to lead trick
            elif weakest_winning_card_is_offsuit:
                return strongest_offsuit
            # Trump in
            else:
                return weakest_winning_card 

        else: # is_last_card
            # Duck if it doesn't matter
            if is_partner_winning or all_cards_would_win or all_cards_would_lose: 
                return weakest_card
            # Win if someone else made bid, or we're on wood, or it's the 4th trick
            elif (not player_made_bid) or (trick_num == 4) or (player.team.tricks_conceded == 2): 
                return weakest_winning_card
            # Duck if I made bid and am in trouble
            elif ((trick_num in [1,2]) and (num_trumps_in_hand in [1,2])) or ((trick_num in [3]) and (num_trumps_in_hand == 1)) and weakest_winning_card_is_trump:
                return weakest_card_bare_offsuit
            # Win otherwise
            else:
                return weakest_winning_card

        # Return random card if error
        return random.choice(valid_cards)

    raise ValueError('Invalid strategy')