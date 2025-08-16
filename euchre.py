import numpy as np
import random

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

from deck import Card, Suit
from participants import Team, Player
from gameplay import Game, Hand, Trick
from strategy import pickup_and_discard, make_bid, select_card

import pandas as pd

bidding_log = pd.DataFrame(columns=[
    "hand_id", "num_trump", "right_bower", "left_bower", "trump_ace", "trump_king",
    "trump_queen", "trump_ten", "trump_nine", "offsuit_aces",
    "flipped_jack", "flipped_ace", "seat_position", "bidding_round",
    "bid_made", "alone", "score"
])

def log_bidding_features(hand_id, player_cards, trumps, flipped_card, seat_position, bidding_round, bid_made, alone):
    global bidding_log
    
    trump_cards = [c for c in player_cards if c.effective_suit(trumps) == trumps]
    num_trump = len(trump_cards)

    log_entry = {
        "hand_id": hand_id,  # link to later scoring
        "num_trump": num_trump,
        "right_bower": int(any(c.is_right_bower(trumps) for c in player_cards)),
        "left_bower": int(any(c.is_left_bower(trumps) for c in player_cards)),
        "trump_ace": sum(c.rank == 'A' and c.effective_suit(trumps) == trumps for c in player_cards),
        "trump_king": sum(c.rank == 'K' and c.effective_suit(trumps) == trumps for c in player_cards),
        "trump_queen": sum(c.rank == 'Q' and c.effective_suit(trumps) == trumps for c in player_cards),
        "trump_ten": sum(c.rank == 'T' and c.effective_suit(trumps) == trumps for c in player_cards),
        "trump_nine": sum(c.rank == '9' and c.effective_suit(trumps) == trumps for c in player_cards),
        "offsuit_aces": sum(c.rank == 'A' and c.effective_suit(trumps) != trumps for c in player_cards),
        "flipped_jack": int(flipped_card.rank == 'J') if bidding_round == 0 else 0,
        "flipped_ace": int(flipped_card.rank == 'A') if bidding_round == 0 else 0,
        "seat_position": seat_position,
        "bidding_round": bidding_round,
        "bid_made": int(bid_made),
        "alone": int(alone),
        "score": None  # placeholder for later
    }
    
    bidding_log = pd.concat([bidding_log, pd.DataFrame([log_entry])], ignore_index=True)

def update_score_for_hand(hand_id, seat_position, score):
    global bidding_log
    idx = (bidding_log['hand_id'] == hand_id) & (bidding_log['seat_position'] == seat_position)
    bidding_log.loc[idx, 'score'] = score


def parse_hand(cards, trumps):
    def rank_rl(card, trumps):
        if card.is_right_bower(trumps):
            return 'R'
        elif card.is_left_bower(trumps):
            return 'L'
        else:
            return card.rank
    
    suits = [card.effective_suit(trumps).abbrev for card in cards]
    values = [card.value_in_trick(trumps, trumps) for card in cards]
    offsuits = list(set(suits)-set(trumps.abbrev))
    d={}
    t = ''.join(rank_rl(card, trumps) for card in cards if card.effective_suit(trumps).abbrev == trumps.abbrev)

    for a in offsuits:
        d[a] = np.dot(sorted([v if s == a else 0 for v, s in zip(values, suits) ], reverse=True), [1e6, 1e3, 1, 0, 0])
    for w in sorted(d, key=d.get, reverse=True):
        t = t + '_' + ''.join(card.rank for card in cards if card.effective_suit(trumps).abbrev == w)

    return t

def init_game(winning_score, player_strategies, player_names):
    """Initializes game of Euchre
    Returns: game, team, players
    """
    
    # Create game
    first_dealer_position = random.randint(0,3)
    game = Game(winning_score, first_dealer_position)
    
    # Create teams
    teams = [Team(0), Team(1)]
    
    # Create players
    players = [Player(0, player_strategies[0], name = player_names[0]),
               Player(1, player_strategies[1], name = player_names[1]), 
               Player(2, player_strategies[2], name = player_names[2]), 
               Player(3, player_strategies[3], name = player_names[3])]
    
    # Assign players into teams
    assign_teams(players, teams)
    
    return game, teams, players

def init_hand(game, teams, players):
    """Initializes hand of Euchre
    Returns: instance of Hand object
    """
    # Reset indicators
    for player in players:
        player.reset_bids()
        
    for team in teams:
        team.reset_trick_count()
        
    # Increment dealer
    game.increment_dealer()
    
    # Create new hand
    hand = Hand(players, game.dealer_position)
    
    return hand

def assign_teams(players, teams):
    """Assign players into teams based on their position. Called at the start of a new game
    """
    for player in players:
        partner_position = np.mod(player.position + 2, 4)
        team_number = np.where(player.position % 2 == 0, 0, 1)
        
        player.partner = players[partner_position]
        player.team = teams[team_number]
        player.opp_team = teams[1-team_number]
        
def create_shuffled_deck(min_rank=9):
    """Returns a shuffled deck of cards
    """

    # Define ranks based on minimum card rank
    suits = 'hdcs'
    all_ranks = '23456789TJQKA'
    assert(str(min_rank) in all_ranks)
    ranks = all_ranks[all_ranks.find(str(min_rank)):]
    
    # Create deck
    deck = []
    for rank in ranks:
        for abbrev in suits:
            deck.append(Card(rank, Suit(abbrev)))

    # Return shuffled deck
    return np.random.permutation(deck)

def deal(players, hand):
    """Randomly deals five cards to each player and flips one over
    """
    
    # Create shuffled deck
    deck = create_shuffled_deck()
   
    # Deal 5 cards to every player
    for p in range(4):
        players[p].cards = list(deck[5*p:5*p+5])
        logging.info('%s: %s', players[p].name, ' '.join([str(c) for c in players[p].cards]))
        
    # Flip one card over
    hand.flipped_card = deck[21]
    logging.info('\nFlipped: %s', hand.flipped_card)

def bidding_round(players, hand, game):
    """Conducts a bidding round over the flipped card
    """
    for bid_round in range(2):
        for table_position in range(4):
            player = hand.bid_order[table_position]
            bid, alone, suit, this_suit = make_bid(player.strategy, player.cards, bid_round, table_position, hand.flipped_card)
            log_bidding_features(
                hand_id=hand.hand_id,
                player_cards=player.cards,
                trumps=this_suit,
                flipped_card=hand.flipped_card,
                seat_position=(player.position - hand.dealer.position) % 4,
                bidding_round=bid_round,
                bid_made=bid,
                alone = alone
            )
            if bid:
                hand.maker = player
                hand.trumps = hand.flipped_card.suit if (bid_round == 0) else suit
                player.is_maker = True
                player.partner.is_partner_maker = True
                player.is_alone = True if (bid_round == 0) and (table_position == 1) else alone
                player.partner.is_partner_alone = player.is_alone
                player.n_bids += 1
                player.n_lone_bids += alone
                if bid_round == 1:
                    hand.bid_type = 'open'
                    player.n_open_bids += 1
                elif table_position in [0, 2]:
                    hand.bid_type = 'order_up'
                    player.n_orderup_bids += 1
                elif table_position == 1:
                    hand.bid_type = 'down_alone'
                    player.n_downalone_bids += 1
                else:
                    hand.bid_type = 'pick_up'
                    player.n_pickup_bids += 1
                    
                logging.info('%s bids %s %s\n', hand.maker.name, hand.trumps.print_symbol(), 'ALONE' if player.is_alone else '')
                
                if bid_round == 0:
                    hand.dealer.cards = pickup_and_discard(hand.dealer.cards, hand.flipped_card)
                    
                for p in players:
                    p.cards.sort(key = lambda x: x.value_in_trick(hand.trumps, hand.trumps), reverse=True)
                    logging.info('%s: %s', p.name, ' '.join([c.__str__(hand.trumps) for c in p.cards]))
                logging.info('')
                
                # Log bid data
                if hand.maker.strategy.startswith('variant'):
                    maker_cards_str = parse_hand(hand.maker.cards, hand.trumps)
                    game.bid_data.append([bid_round, table_position, hand.flipped_card.rank, maker_cards_str, 1 if hand.maker.is_alone else 0])
            
                return
            
            logging.info('%s passes', player.name)
    
def playing_round(players, teams, hand, game):
    """Conducts a playing round after bids have been made
    """
    
    # Set leader of first trick
    leader_position = hand.bid_position_order[0]
    
    for trick_num in range(5):
        # Play trick
        trick = Trick(hand, players, leader_position)
        trick.play()
        
        # Update trick count
        trick.winner.team.tricks_taken += 1
        trick.winner.opp_team.tricks_conceded += 1
        logging.info('%s leads: %s %s takes (%s-%s)', players[leader_position].name, ' '.join([c.__str__(hand.trumps) for c in trick.cards_played]), trick.winner.name, teams[0].tricks_taken, teams[1].tricks_taken)
        
        # Set leader of next trick
        leader_position = trick.winner.position
        
    # Update game score based on result of the playing round
    hand.winner = teams[0] if teams[0].tricks_taken >= 3 else teams[1]
    res = hand.result()
    pts = hand.winner_points()
    maker_pts = -pts if res == 'euchre' else pts
    hand.winner.game_score += pts
    
    if res == 'lone_march':
        hand.maker.n_lone_marches += 1
    elif res == 'march':
        hand.maker.n_marches += 1
    elif res == 'euchre':
        hand.maker.n_euchred += 1
    else:
        hand.maker.n_singles += 1
        
    hand.maker.bid_score += maker_pts
    hand.maker.lone_bid_score += maker_pts if hand.maker.is_alone else 0
    hand.maker.pickup_bid_score += maker_pts if hand.bid_type == 'pick_up' else 0
    hand.maker.downalone_bid_score += maker_pts if hand.bid_type == 'down_alone' else 0
    hand.maker.orderup_bid_score += maker_pts if hand.bid_type == 'order_up' else 0
    hand.maker.open_bid_score += maker_pts if hand.bid_type == 'open' else 0
    
    for seat_position in range(4):
        if hand.winner == teams[0]:
            update_score_for_hand(hand.hand_id, seat_position, (1-2*((seat_position + hand.dealer.position) % 2))*pts)
        else:
            update_score_for_hand(hand.hand_id, seat_position, (2*((seat_position + hand.dealer.position) % 2)-1)*pts)
    
    # Log bid data
    if hand.maker.strategy.startswith('variant'):
        game.bid_data[-1].append(maker_pts)
    
    # Log experiment
    for player in players:
        if player.experiment_triggered:
            player_pts = maker_pts if player.team == hand.maker.team else -maker_pts
            player.experiment_score += player_pts
        
        
    logging.info('\n%s wins the round (%s-%s)', hand.winner.name, teams[0].game_score, teams[1].game_score)
    results.append({
        "round": hand.hand_id,  # however you track rounds
        "winner": hand.winner.name,
        "team0_score": teams[0].game_score,
        "team1_score": teams[1].game_score
    })

results = []
def play_euchre(winning_score, player_names = [None] * 4, player_strategies = ['scorecard_complex'] * 4, verbose=False):
    """Plays a game of Euchre
    """
    logger = logging.getLogger()
    logger.disabled = not verbose
    game, teams, players = init_game(winning_score, player_strategies, player_names)

    while game.winner(teams) is None:
        # Start a new hand
        hand = init_hand(game, teams, players)
        logging.info('%s', '-'*100)
        logging.info('Dealer: %s', hand.dealer.name)

        # Deal
        deal(players, hand)
        game.n_hands += 1

        # Conduct bidding round
        bidding_round(players, hand, game)

        # Toss in if no bids
        if hand.maker == None:
            logging.info('Toss in')
            continue

        # Conduct playing round
        playing_round(players, teams, hand, game)

    bidding_log.to_csv("bidding_data.csv", index=False)
    
    logging.info('%s', '-'*100)
    logging.info('%s wins!', game.winner(teams).name)
    pd.DataFrame(results).to_csv("game_scores.csv", index=False)
    # return score_json(game, teams, players)

def score_json(game, teams, players):
    """Returns a json of the result
    """
    game_keys = ['winning_score', 'first_dealer_position', 'n_hands']
    team_keys = ['number', 'name', 'game_score']
    player_keys = ['position', 'name', 'team', 'opp_team', 'partner', 'strategy', 'n_bids', 'n_lone_bids', 'n_pickup_bids', 'n_orderup_bids', 'n_downalone_bids', 'n_open_bids', 'n_singles', 'n_marches', 'n_lone_marches', 'n_euchred', 'bid_score', 'lone_bid_score', 'pickup_bid_score', 'orderup_bid_score', 'downalone_bid_score', 'open_bid_score', 'n_experiments', 'experiment_score']

    d = {k: vars(game)[k] for k in game_keys if k in vars(game)}
    d['winner'] = game.winner(teams)
    d['teams'] = []
    for team in teams:
        d['teams'].append({k: vars(team)[k] for k in team_keys if k in vars(team)})
    d['players'] = []
    for player in players:
        d['players'].append({k: vars(player)[k] for k in player_keys if k in vars(player)})
    
    return d