from colorama import Back, Fore

class Suit:
    def __init__(self, abbrev):
        self.abbrev = abbrev
        self.symbol = self.to_symbol()
        self.color = self.to_color()
        
    def __key(self):
        return (self.abbrev)
    
    def __hash__(self):
        return hash(self.__key())
    
    def __eq__(self, other):
        return (self.abbrev == other.abbrev)
    
    def to_symbol(self):
        suit_lookup = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
        return suit_lookup[self.abbrev] if self.abbrev in suit_lookup else None
   
    def to_color(self):
        color_lookup = {'h': 'r', 'd': 'r', 'c': 'b', 's': 'b'}
        return color_lookup[self.abbrev] if self.abbrev in color_lookup else None
    
    def print_symbol(self):
        fore = Fore.RED if self.color == 'r' else Fore.BLACK
        return fore + self.symbol + Fore.RESET

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.ordinal = self.rank_to_ordinal()
    
    def __key(self):
        return (self.rank, self.suit)
    
    def __hash__(self):
        return hash(self.__key())
    
    def __str__(self, trumps=None):
        return self.print_card(trumps)
    
    def __eq__(self, other):
        return (self.rank == other.rank) and (self.suit == other.suit)
    
    def print_card(self, trumps=None):
        fore = Fore.RED if self.suit.color == 'r' else Fore.BLACK
        back = Back.WHITE if (trumps is not None) and (self.effective_suit(trumps) == trumps) else ''
        return fore + back + self.rank + self.suit.symbol + Fore.RESET + Back.RESET
   
    def rank_to_ordinal(self):
        ordinal_lookup = '23456789TJQKA'
        return ordinal_lookup.find(self.rank) + 2 if self.rank in ordinal_lookup else None
       
    def is_trump_suit(self, trumps):
        return (self.suit == trumps)
   
    def is_right_bower(self, trumps):
        return self.is_trump_suit(trumps) and (self.rank == 'J')
   
    def is_left_bower(self, trumps):
        return (not self.is_trump_suit(trumps)) and (self.suit.color == trumps.color) and (self.rank == 'J')
    
    def is_bower(self, trumps):
        return (self.suit.color == trumps.color) and (self.rank == 'J')
   
    def is_offsuit_ace(self, trumps):
        return (not self.is_trump_suit(trumps)) and (self.rank == 'A')
    
    def effective_suit(self, trumps):
        #if self.is_trump_suit(trumps) or self.is_left_bower(trumps):
        if self.is_bower(trumps):
            return trumps
        else:
            return self.suit
        
    def value_in_trick(self, trumps, suit_led):
        """Returns the value of a card in the trick, based on trump suits and the suit led
        """
        if self.is_right_bower(trumps):
            return 216
        elif self.is_bower(trumps):
            return 215
        elif self.is_trump_suit(trumps):
            return 200 + self.ordinal
        elif self.suit == suit_led:
            return 100 + self.ordinal
        else:
            return self.ordinal # used to determine card to throw away, even though of equal value