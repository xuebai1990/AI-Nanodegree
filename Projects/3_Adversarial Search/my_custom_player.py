
from sample_players import DataPlayer
import random

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
#        if state.board in self.data:
#            self.queue.put(self.data[state.board])
#            return
        self.queue.put(random.choice(state.actions()))
        self.depth = 1
        while True:
            self.queue.put(self.decision(state, self.depth))
            self.depth += 1

    def decision(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in state.actions():
            v = self.min_value(state.result(action), alpha, beta, depth - 1)
            if v > best_score:
                best_score = v
                best_move = action
            alpha = max(alpha, v)
        return best_move if best_move else state.actions()[0]

    def min_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.heuristic(state)
        v = float("inf")
        actions = state.actions()
#        actions.sort(key=lambda x: self.nummove(state.result(x), 1 - self.player_id), reverse=True)
        for action in actions:
            v = min(v, self.max_value(state.result(action), alpha, beta, depth - 1))
            if v <= alpha: return v
            beta = min(beta, v)
        return v

    def max_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.heuristic(state)
        v = float("-inf")
        actions = state.actions()
#        actions.sort(key=lambda x: self.nummove(state.result(x), self.player_id), reverse=True)
        for action in state.actions():
            v = max(v, self.min_value(state.result(action), alpha, beta, depth - 1))
            if v >= beta: return v
            alpha = max(alpha, v)
        return v

    # To call the heuristic function
    def heuristic(self, state):
#        my_move = self.nummove(state, self.player_id)
#        oppo_move = self.nummove(state, 1 - self.player_id)
        my_count = self.liberty(state, self.player_id)
        oppo_count = self.liberty(state, 1 - self.player_id)
#        return my_move - oppo_move
        return my_count - oppo_count

    # Number of open cells available withon a 5*5 squre
    def liberty(self, state, player):
        if not state.locs[player]: return 24
        width = 11
        S, N, W, E = -width - 2, width + 2, 1, -1
        actions = (S, N, W, E, N + W, N + E, S + W, S + E, \
                   2*N+2*W, 2*N+W, 2*N, 2*N+E, 2*N+2*E, N+2*W, N+2*E, 2*W, 2*E, S+2*W, S+2*E, 2*S+2*W, 2*S+W, 2*S, 2*S+E, 2*S+2*E)
        count = 0
        for action in actions:
            if (action + state.locs[player]) > 0 and (state.board & (1 << (action + state.locs[player]))):
                count += 1
        return count

    # Number of available next moves
    def nummove(self, state, player):
        return len(state.liberties(state.locs[player]))

    # Number of open cells available within a 3*3 squre
    def liberty2(self, state, player):
        if not state.locs[player]: return 8
        width = 11
        S, N, W, E = -width - 2, width + 2, 1, -1
        actions = (S, N, W, E, N + W, N + E, S + W, S + E)
        count = 0
        for action in actions:
            if (action + state.locs[player]) > 0 and (state.board & (1 << (action + state.locs[player]))):
                count += 1
        return count

