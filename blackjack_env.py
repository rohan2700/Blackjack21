import numpy as np
import random
from collections import defaultdict
from enum import Enum

# Define actions available to the player
class Action(Enum):
    STAND = 0
    HIT = 1
    DOUBLE = 2
    SPLIT = 3
    SURRENDER = 4  # New action: Player can surrender their hand

class BlackjackEnv:
    """
    A simplified Blackjack environment for reinforcement learning.
    Supports various rule variations.
    """
    def __init__(self, decks=6, penetration=0.5, rules=None):
        """
        Initializes the Blackjack environment.

        Args:
            decks (int): Number of decks in the shoe.
            penetration (float): Proportion of the shoe to be played before reshuffling.
            rules (set): A set of strings representing enabled rule variations (e.g., {'double_allowed', 'late_surrender'}).
        """
        self.decks = decks
        self.penetration = penetration
        self.rules = rules if rules else set()  # Store rules as a set for easy lookup
        self.action_space = len(Action)  # Total number of possible actions
        self.gamma = 0.95  # Discount factor (though often 1.0 for episodic tasks like Blackjack)
        self.action_meanings = ["STAND", "HIT", "DOUBLE", "SPLIT", "SURRENDER"]
        self.initial_num_decks = decks
        self.cards_drawn_this_episode = 0
        self.done = False  # Flag to indicate if the episode is finished
        self.deck = []  # The current shoe of cards
        self.cut_card = 0  # Index where reshuffling occurs

    def reset(self):
        """
        Resets the environment for a new episode (hand).
        Creates a new shoe, deals initial cards, and resets episode state.

        Returns:
            tuple: The initial state of the environment (player_value, dealer_up_card_value, usable_ace).
        """
        self.player_hand = []
        self.dealer_hand = []
        self.deck = self._create_shoe()  # Create and shuffle a new shoe
        self.done = False
        self.cards_drawn_this_episode = 0
        self._deal_initial_cards()  # Deal two cards to player and dealer
        return self._get_state()

    def _create_shoe(self):
        """
        Creates a new shoe (deck(s) of cards) and shuffles it.
        Sets the cut card position based on penetration.

        Returns:
            list: A list representing the shuffled shoe.
        """
        ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4  # 10 for J, Q, K; 11 for Ace
        shoe = ranks * self.decks
        random.shuffle(shoe)
        # Cut card ensures a certain percentage of cards are played before reshuffling
        self.cut_card = max(1, int(len(shoe) * self.penetration))
        return shoe

    def _draw_card(self):
        """
        Draws a single card from the shoe. If the cut card is reached, reshuffles.

        Returns:
            int: The value of the drawn card. Returns 0 if deck is empty (should not happen with proper logic).
        """
        self.cards_drawn_this_episode += 1
        if len(self.deck) <= self.cut_card:
            # Reshuffle if penetration reached.
            # For counting env, running_count reset happens in its _draw_card override.
            self.deck = self._create_shoe()

        if not self.deck:
            # Fallback for an empty deck, though ideally, this condition is avoided.
            return 0

        card = self.deck.pop()
        return card

    def _deal_initial_cards(self):
        """Deals two cards to the player and two to the dealer."""
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]

    def _hand_value(self, hand):
        """
        Calculates the value of a given hand, handling Aces (1 or 11).

        Args:
            hand (list): A list of card values in the hand.

        Returns:
            int: The calculated total value of the hand.
        """
        total = sum(hand)
        aces = hand.count(11)

        # Convert Ace from 11 to 1 if busting (total > 21)
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    def _resolve_hand(self):
        """
        Resolves the outcome of the hand after the player stands or busts.
        Dealer plays their hand according to standard rules (hit until 17+).

        Returns:
            int: Reward for the hand (1 for win, 0 for push, -1 for loss).
        """
        player_val = self._hand_value(self.player_hand)
        dealer_val = self._hand_value(self.dealer_hand)

        # NEW Rule: Player 21 always wins (even against dealer's 21)
        if 'player_21_always_wins' in self.rules and player_val == 21:
            return 1  # Player wins if they have 21, regardless of dealer's hand

        if player_val > 21:
            return -1  # Player busts, loses

        # Dealer hits until their hand value is 17 or more
        while dealer_val < 17:
            self.dealer_hand.append(self._draw_card())
            dealer_val = self._hand_value(self.dealer_hand)

        if dealer_val > 21:
            return 1  # Dealer busts, player wins
        elif player_val > dealer_val:
            return 1  # Player has higher value, player wins
        elif player_val < dealer_val:
            return -1  # Dealer has higher value, player loses
        return 0  # It's a push (tie)

    def _get_state(self):
        """
        Returns the current state of the environment from the player's perspective.

        Returns:
            tuple: (player_hand_value, dealer_up_card_value, usable_ace_indicator).
        """
        player_val = self._hand_value(self.player_hand)
        dealer_val = self._hand_value([self.dealer_hand[0]])  # Only dealer's up-card is visible
        usable_ace = 1 if 11 in self.player_hand and player_val <= 21 else 0  # 1 if player has a soft hand, 0 otherwise
        return (player_val, dealer_val, usable_ace)

    def step(self, action):
        """
        Processes a single action taken by the player.

        Args:
            action (int): The action chosen by the agent (from Action Enum).

        Returns:
            tuple: (next_state, reward, done, info_dict).
                next_state (tuple): The new state after the action.
                reward (float): The immediate reward received.
                done (bool): True if the episode has ended, False otherwise.
                info_dict (dict): Additional information (e.g., original action taken).
        """
        if self.done:
            # If the episode is already done, no further actions are allowed.
            return self._get_state(), 0, True, {}

        reward = 0
        current_done_state = False
        original_action = action  # Store the action requested by the agent

        # Check if the dealer has a natural blackjack (used for late surrender rule)
        # This is a simplified check; in real casinos, dealer peeks for BJ only on 10/Ace up-card.
        # Here, we check the full initial dealer hand.
        dealer_has_natural_blackjack = (len(self.dealer_hand) == 2 and self._hand_value(self.dealer_hand) == 21)

        # Handle SURRENDER action
        if action == Action.SURRENDER.value:
            # Late surrender rule: allowed only on initial two cards and if dealer does NOT have natural BJ
            if 'late_surrender' in self.rules and len(self.player_hand) == 2:
                if not dealer_has_natural_blackjack:
                    reward = -0.5  # Player gets half their bet back
                    current_done_state = True
                else:
                    # Dealer has natural blackjack, surrender is invalid.
                    # The agent's surrender action is ignored, and they must play on.
                    # We'll treat this as if they chose to STAND as a default invalid action.
                    action = Action.STAND.value
            else:
                # Surrender not allowed (e.g., rule not enabled, or not initial hand).
                # Treat as STAND as a fallback for an invalid action.
                action = Action.STAND.value

        # Proceed with other actions if the episode is not already done by surrender
        if not current_done_state:
            if action == Action.HIT.value:
                self.player_hand.append(self._draw_card())
                if self._hand_value(self.player_hand) > 21:
                    current_done_state = True
                    reward = -1  # Player busts

            elif action == Action.STAND.value:
                current_done_state = True
                reward = self._resolve_hand()

            elif action == Action.DOUBLE.value and 'double_allowed' in self.rules:
                if len(self.player_hand) == 2:  # Can only double down on initial two cards
                    self.player_hand.append(self._draw_card())
                    current_done_state = True
                    if self._hand_value(self.player_hand) > 21:
                        reward = -2  # Double down and bust (lose twice the bet)
                    else:
                        reward = self._resolve_hand() * 2  # Double down and resolve (win/lose twice)
                else:
                    # Invalid double attempt (not initial hand), treat as STAND
                    action = Action.STAND.value
                    current_done_state = True
                    reward = self._resolve_hand()

            elif action == Action.SPLIT.value and 'splitting_allowed' in self.rules and \
                 len(self.player_hand) == 2 and self.player_hand[0] == self.player_hand[1]:
                # Simplified splitting: For this environment, we'll treat a split action
                # as a STAND. A full splitting implementation would require managing
                # multiple hands and is significantly more complex for this structure.
                action = Action.STAND.value
                current_done_state = True
                reward = self._resolve_hand()
            else:
                # Fallback for any other invalid action or disabled rule: treat as STAND
                action = Action.STAND.value
                current_done_state = True
                reward = self._resolve_hand()

        self.done = current_done_state
        # Return original_action in info for debugging/tracking purposes
        return self._get_state(), reward, self.done, {'action_taken': original_action}

    # Helper method to resolve a single player hand against a given dealer hand
    def _resolve_one_hand(self, player_current_hand_value, dealer_current_hand_list):
        if player_current_hand_value > 21:
            return -1 # Player busts

        local_dealer_hand = list(dealer_current_hand_list) # Work on a copy of dealer's hand for this resolution

        # Dealer hits until 17 or more
        while self._hand_value(local_dealer_hand) < 17:
            local_dealer_hand.append(self._draw_card()) # Draws from the shared deck

        dealer_val = self._hand_value(local_dealer_hand)

        if dealer_val > 21:
            return 1 # Dealer busts, player wins
        elif player_current_hand_value > dealer_val:
            return 1 # Player has higher value, player wins
        elif player_current_hand_value < dealer_val:
            return -1 # Dealer has higher value, player loses
        return 0 # It's a push (tie)


class BlackjackWithCountingEnv(BlackjackEnv):
    """
    Extends BlackjackEnv to include card counting functionality.
    The state space is augmented with a 'true count' bin.
    """
    def __init__(self, decks=6, penetration=0.5, rules=None):
        super().__init__(decks, penetration, rules)
        self.card_values = self._create_count_system()  # Defines card values for counting
        self.running_count = 0  # The current running count
        self.min_bet = 1  # Minimum bet for betting strategy
        self.max_bet = 10  # Maximum bet for betting strategy
        self.initial_deck_size = 52 * self.decks

    def _create_count_system(self):
        """
        Defines the High-Low card counting system values.
        """
        return {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1}

    def _draw_card(self):
        """
        Overrides the base _draw_card to update the running count.
        Resets running count when a new shoe is created.
        """
        self.cards_drawn_this_episode += 1
        if len(self.deck) <= self.cut_card:
            self.deck = self._create_shoe()
            self.running_count = 0  # Reset count when a new shoe is introduced

        if not self.deck:
            return 0

        card = self.deck.pop()
        self.running_count += self.card_values.get(card, 0)  # Update running count
        return card

    def reset(self):
        """
        Resets the environment and the running count for a new hand.
        """
        self.running_count = 0  # Reset running count at the start of each new hand/episode
        return super().reset()

    def _get_state(self):
        """
        Overrides the base _get_state to include the binned true count.

        Returns:
            tuple: (player_value, dealer_up_card_value, usable_ace_indicator, true_count_bin).
        """
        base_state = super()._get_state()

        deck_remaining_cards = len(self.deck)
        # Calculate effective decks remaining, with a minimum to prevent division by zero
        effective_decks_remaining = deck_remaining_cards / 52.0
        if effective_decks_remaining < 0.1:
            effective_decks_remaining = 0.1

        true_count = self.running_count / effective_decks_remaining

        # Bin the true count into discrete categories for state representation
        if true_count < -3:
            count_bin = -2
        elif true_count < -1:
            count_bin = -1
        elif true_count < 1:
            count_bin = 0
        elif true_count < 3:
            count_bin = 1
        else:
            count_bin = 2

        return (*base_state, count_bin)

    def step(self, action):
        """
        Processes a single action taken by the player, applying betting strategy.

        Args:
            action (int): The action chosen by the agent (from Action Enum).

        Returns:
            tuple: (next_state, reward, done, info_dict).
                next_state (tuple): The new state after the action.
                reward (float): The immediate reward received.
                done (bool): True if the episode has ended, False otherwise.
                info_dict (dict): Additional information (e.g., original action taken).
        """
        if self.done:
            return self._get_state(), 0, True, {}

        reward = 0
        current_done_state = False
        original_action = action

        dealer_has_natural_blackjack = (len(self.dealer_hand) == 2 and self._hand_value(self.dealer_hand) == 21)

        if action == Action.SURRENDER.value:
            if 'late_surrender' in self.rules and len(self.player_hand) == 2:
                if not dealer_has_natural_blackjack:
                    reward = -0.5
                    current_done_state = True
                else:
                    action = Action.STAND.value # Treat as STAND if surrender is invalid
            else:
                action = Action.STAND.value # Treat as STAND if surrender not allowed by rules/hand

        if not current_done_state:
            if action == Action.HIT.value:
                self.player_hand.append(self._draw_card())
                if self._hand_value(self.player_hand) > 21:
                    current_done_state = True
                    reward = -1

            elif action == Action.STAND.value:
                current_done_state = True
                reward = self._resolve_hand()

            elif action == Action.DOUBLE.value and 'double_allowed' in self.rules:
                if len(self.player_hand) == 2:
                    self.player_hand.append(self._draw_card())
                    current_done_state = True
                    if self._hand_value(self.player_hand) > 21:
                        reward = -2
                    else:
                        reward = self._resolve_hand() * 2
                else:
                    # Invalid double attempt (not initial hand), treat as STAND
                    action = Action.STAND.value
                    current_done_state = True
                    reward = self._resolve_hand()

            elif action == Action.SPLIT.value and 'splitting_allowed' in self.rules and \
                 len(self.player_hand) == 2 and self.player_hand[0] == self.player_hand[1]:

                card_to_split = self.player_hand[0]
                total_split_reward = 0
                original_dealer_hand_at_split = list(self.dealer_hand) # Capture dealer's initial hand for both resolutions

                # --- Play out First Split Hand ---
                current_player_split_hand_1 = [card_to_split, self._draw_card()]
                # Simplified strategy for split hands: hit until 17 or bust
                while self._hand_value(current_player_split_hand_1) < 17 and self._hand_value(current_player_split_hand_1) <= 21:
                    current_player_split_hand_1.append(self._draw_card())

                # Resolve this split hand against the dealer's original hand
                reward1 = self._resolve_one_hand(self._hand_value(current_player_split_hand_1), original_dealer_hand_at_split)
                total_split_reward += reward1

                # --- Play out Second Split Hand ---
                current_player_split_hand_2 = [card_to_split, self._draw_card()]
                # Simplified strategy for split hands: hit until 17 or bust
                while self._hand_value(current_player_split_hand_2) < 17 and self._hand_value(current_player_split_hand_2) <= 21:
                    current_player_split_hand_2.append(self._draw_card())

                # Resolve this split hand against the dealer's original hand
                reward2 = self._resolve_one_hand(self._hand_value(current_player_split_hand_2), original_dealer_hand_at_split)
                total_split_reward += reward2

                reward = total_split_reward # Sum of rewards from both split hands
                current_done_state = True # Splitting always ends the player's turn for the main hand

            else:
                # Fallback for any other invalid action or disabled rule: treat as STAND
                action = Action.STAND.value
                current_done_state = True
                reward = self._resolve_hand()

        self.done = current_done_state

        # Apply betting multiplier if the hand is resolved
        if self.done:
            current_state_tuple_after_action = self._get_state()
            count_bin = current_state_tuple_after_action[3]

            approx_true_count = 0
            if count_bin == -2: approx_true_count = -4
            elif count_bin == -1: approx_true_count = -2
            elif count_bin == 0: approx_true_count = 0
            elif count_bin == 1: approx_true_count = 2
            else: approx_true_count = 4

            bet_multiplier = self.min_bet # Default bet for negative/neutral counts

            # More aggressive betting strategy for positive true counts
            if approx_true_count == 2: # Corresponds to count_bin 1
                bet_multiplier = 4 # Bet 4 units
            elif approx_true_count == 4: # Corresponds to count_bin 2
                bet_multiplier = 8 # Bet 8 units

            bet_multiplier = min(self.max_bet, bet_multiplier) # Cap at max_bet

            reward *= bet_multiplier # Scale reward by the bet multiplier

        final_state = self._get_state()
        return final_state, reward, self.done, {'action_taken': original_action}
