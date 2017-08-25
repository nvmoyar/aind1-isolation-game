"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def manhattan_distance(start, end):
    """Calculates the Manhattan distance between two points. This function is used as helper
    for custom heuristics functions. 
    
    Parameters
    ----------
    start : a single point (0,0) (from)
    end : a single point (6,6) (to)

    Returns
    -------
    int 
        Manhattan distance between two points of the board/grid
    """
    sx, sy = start
    ex, ey = end

    return int(abs(ex - sx) + abs(ey - sy))  


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!

    if game.is_loser(player): 
        return float('-inf')

    if game.is_winner(player): 
        return float('inf')   

      
    n_agent_moves   = len(game.get_legal_moves(player))
    n_opp_moves     = len(game.get_legal_moves(game.get_opponent(player)))
    # n_blank_spaces  = len(game.get_blank_spaces())
    # pcent_blank_spaces = int(n_blank_spaces / (game.width * game.height) * 100) 


    player_loc  = game.get_player_location(player)
    opp_loc     = game.get_player_location(game.get_opponent(player))
  
    # penalizes short positions     
    return float((manhattan_distance(player_loc, opp_loc)/12.0) * len(game.get_legal_moves(player)))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    if game.is_loser(player): 
        return float('-inf')

    if game.is_winner(player): 
        return float('inf')   

    n_agent_moves = len(game.get_legal_moves(player))
    n_opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # penalizes opponent's move in a factor of abitrary number
    return float(n_agent_moves - 4.0 * n_opp_moves)      


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    if game.is_loser(player): 
        return float('-inf')

    if game.is_winner(player): 
        return float('inf')   

      
    n_agent_moves   = len(game.get_legal_moves(player))
    n_opp_moves     = len(game.get_legal_moves(game.get_opponent(player)))
    
    # n_blank_spaces  = len(game.get_blank_spaces())
    # pcent_blank_spaces = int(n_blank_spaces / (game.width * game.height) * 100) 


    player_loc  = game.get_player_location(player)
    opp_loc     = game.get_player_location(game.get_opponent(player))
  
    # longer manhattan distance from the opponent, more penalization to the opponents move    
    return float(n_agent_moves - manhattan_distance(player_loc, opp_loc) * n_opp_moves)       



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=15.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def check_time_left(self):
        """
        Timer function, checks the remaining time and raises exception if the time left is under the TIMER_THRESHOLD 
        As this function is used in both classes Minimax and Alphabeta, it could be included in the IsolationPlayer class.

        Parameters
        ----------
        self : current object
        """    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        legal_moves = game.get_legal_moves()

        if len(legal_moves) == 0:
            return (-1, -1)
        
        # best_move = (-1, -1)
        # best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]
        best_move = legal_moves[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        

        # TODO: finish this function!
                   
        # nvmoyar: Minimax decision 

        self.check_time_left()

        legal_moves = game.get_legal_moves()

        if len(legal_moves) == 0:
            return (-1, -1)
         
        # best_move = (-1, -1) 
        # best_move = random.choice(legal_moves)
        # best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]
        best_move = legal_moves[0]

        # We call min_value() first instead of max_value() because the root node itself is a "max" node    
        best_moves = [(self.min_value(game.forecast_move(move), depth-1), move) for move in legal_moves]
        _value, best_move = max(best_moves)

        return best_move

    # ---> START mutually recursive helper functions  

    def max_value(self, game, depth):
        """
        Helper function to get the max_value for minimax function, given a fixed depth
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        best_score : float 

        """
        
        self.check_time_left()

        # Return a list of all legal moves available to the active player.
        legal_moves = game.get_legal_moves()

        # Check if depth limit has been reached or if the game is in terminal state, then return utility_value
        if len(legal_moves) ==0 or depth == 0:
            return self.score(game, self)

        best_score = float('-inf')

        for move in legal_moves:
            # Return a new board object with the specified move applied to the current game state.
            # board = game.forecast_move(move)
            # best_score is between max of the previous value or the current value returned by the minimax min function
            best_score = max(best_score, self.min_value(game.forecast_move(move), depth-1))

        return best_score

    def min_value(self, game, depth):
        """
        Helper function to get the min_value for minimax function given a fixed depth

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        best_score : float 

        """
        
        self.check_time_left()

        # Return a list of all legal moves available to the active player.
        legal_moves = game.get_legal_moves()

        if len(legal_moves) ==0 or depth == 0:
            return self.score(game, self)

        best_score = float('inf')

        for move in legal_moves:
            best_score = min(best_score, self.max_value(game.forecast_move(move), depth-1))

        return best_score    

    # ---> END mutually recursive helper functions       


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def check_time_left(self):
        """
        Timer function, checks the remaining time and raises exception if the time left is under the TIMER_THRESHOLD 
        As this function is used in both classes Minimax and Alphabeta, it could be included in the IsolationPlayer class.
         
        Parameters
        ----------
        self : current object
        """    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        
        # nvmoyar: 

        legal_moves = game.get_legal_moves()

        if len(legal_moves) == 0: 
            return (-1, -1)

        # best_move = (-1, -1)    
        # best_move = random.choice(legal_moves)
        # best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]
        best_move = legal_moves[0]

        try: 
            depth =  1
            while True : # infinite loop till end of game, exploring one deeper level at a time
                best_move = self.alphabeta(game, depth)
                depth +=1

        except SearchTimeout:
            pass 
        
        return best_move            

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        # TODO: finish this function!

        # nvmoyar: Minimax decision + alphabeta pruning 
    
        self.check_time_left()
        legal_moves = game.get_legal_moves()

        # best_move = (-1, -1)
        # best_move = random.choice(legal_moves)
        # best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]
        best_move = legal_moves[0]

        best_score=float('-inf')
             
        for move in legal_moves:
            new_board = game.forecast_move(move)

            # We call min_value() first instead of max_value() because the root node itself is a "max" node
            # get the score for the current branch
            _value = self.min_value(new_board, alpha, beta, depth-1)
            # check if the score is better than the current score (the last updated)
            if _value > best_score: # the score is better, update score and move
                best_move = move
                best_score = _value

            # update the lower bound of search and keep on searching at the current depth    
            alpha=max(best_score, alpha)

        return best_move

    # ---> START mutually recursive helper functions    

    def max_value(self, game, alpha, beta, depth):
        """
        Helper function to get the max_value for minimax search with alpha-beta pruning
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        best_score : float 

        """

        self.check_time_left()
        legal_moves = game.get_legal_moves()

        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, self)
        
        _value = float('-inf')
        
        for move in legal_moves:
            new_board = game.forecast_move(move)
           
            min_v = self.min_value(new_board, alpha, beta, depth-1)
            _value = max(_value, min_v)
           
            if _value >= beta: # if _value is lower than beta (the upper bound) update the lower bound of search (alpha)
                return _value
            alpha = max(_value, alpha)

        return _value

    def min_value(self, game, alpha, beta, depth):
        """
        Helper function to get the min_value for minimax search with alpha-beta pruning
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        best_score : float 

        """
        
        self.check_time_left()
        legal_moves = game.get_legal_moves()

        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, self)
               
        _value = float('inf')

        for move in legal_moves:
            new_board=game.forecast_move(move)
            max_v = self.max_value(new_board, alpha, beta, depth-1)
            _value = min(_value, max_v)

            if _value <= alpha:
                return _value
            beta = min(beta, _value)

        return _value

      # ---> END mutually recursive helper functions     