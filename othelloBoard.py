"""
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1 = white, -1 = black, 0 = empty
  first dim is column, 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples. 
x is the column, y is the row.

https://github.com/hylbyj/Alpha-Beta-Pruning-for-Othello-Game/blob/master/board.py
"""

class Board():
    # List of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self):
        """ Set up initial board configuration. """
        # Create the empty board array
        self.pieces = [None]*8
        for i in range(8):
            self.pieces[i] = [0]*8

        # Set up the initial 4 pieces
        self.pieces[3][4] = 1
        self.pieces[4][3] = 1
        self.pieces[3][3] = -1;
        self.pieces[4][4] = -1;

    # Add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def display(self):
        """" Display the board and the statistics of the ongoing game. """
        print("    A B C D E F G H")
        print("    ---------------")
        for y in range(8):
            # Print the row number
            row = str(y+1) + ' |'
            for x in range(8):
                # Get the piece to print
                piece = self[x][y]
                if piece == -1:
                    row += "B "
                elif piece == 1:
                    row += "W "
                else:
                    row += ". "
            
            row += '| ' + str(y+1)
            print(row)
        print("    ---------------")
        print("    A B C D E F G H\n")

        print("STATISTICS (score)")
        print("Black: " + str(self.count(-1)))
        print("White: " + str(self.count(1)))

    def count(self, color):
        """ Count the number of pieces of the given color.
        (1 for white, -1 for black, 0 for empty spaces) """
        count = 0
        for y in range(8):
            for x in range(8):
                if self[x][y]==color:
                    count += 1
        return count

    def get_squares(self, color):
        """ Get the coordinates (x,y) for all pieces on the board of the given color.
        (1 for white, -1 for black, 0 for empty spaces) """
        squares=[]
        for y in range(8):
            for x in range(8):
                if self[x][y]==color:
                    squares.append((x,y))
        return squares


    def get_legal_moves(self, color):
        """ Return all the legal moves for the given color.
        (1 for white, -1 for black) """
        # Store the legal moves
        moves = set()
        # Get all the squares with pieces of the given color.
        for square in self.get_squares(color):
            # Find all moves using these pieces as base squares.
            newmoves = self.get_moves_for_square(square)
            # Store these in the moves set.
            moves.update(newmoves)
        return list(moves)


    def get_moves_for_square(self, square):
        """ Return all the legal moves that use the given square as a base 
        square. That is, if the given square is (3,4) and it contains a black 
        piece, and (3,5) and (3,6) contain white pieces, and (3,7) is empty, 
        one of the returned moves is (3,7) because everything from there to 
        (3,4) can be flipped. """
        (x,y) = square
        # Determine the color of the piece
        color = self[x][y]

        # Skip empty source squares
        if color==0:
            return None

        # Search all possible directions
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                moves.append(move)
        # Return the generated list of moves
        return moves

    def execute_move(self, move, color):
        """ Perform the given move on the board, and flips pieces as necessary.
        color gives the color of the piece to play (1 for white, -1 for black) """
        # Start at the new piece's square and follow it on all 8 directions
        # to look for pieces allowing flipping

        # Add the piece to the empty square
        flips = (flip for direction in self.__directions
                      for flip in self._get_flips(move, direction, color))

        for x,y in flips:
            self[x][y] = color

    def _discover_move(self, origin, direction):
        """ Return the endpoint of a legal move, starting at the given origin,
        and moving in the given direction. """
        x,y = origin
        color = self[x][y]
        flips = False

        #for j in Board._increment_move(origin, direction):
            #print(j)

        for x,y in Board._increment_move(origin, direction):
            if self[x][y] == 0 and flips:
                return (x,y)
            elif (self[x][y] == color or (self[x][y] == 0 and not flips)):
                return None
            elif self[x][y] == -color:
                flips = True

    def _get_flips(self, origin, direction, color):
        """ Get the list of flips for a vertex and a direction to use within 
        the execute_move function. """
        # Initialize variable
        flips = [origin]

        for x, y in Board._increment_move(origin, direction):
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == 0:
                break
            elif self[x][y] == color and len(flips) > 1:
                return flips
        return []

    @staticmethod
    def _increment_move(origin, direction):
        move = []
        
        d = list(direction)
        for i in range(8):
            move.append(list(d))
            d[0] += direction[0]
            d[1] += direction[1]

        for i in range(8):
            move[i][0] += origin[0]
            move[i][1] += origin[1]

        return move
             
        
        #""" Generator expression for incrementing moves """
        #move = map(sum, zip(move, direction))
        #while all(map(lambda x: 0 <= x < 8, move)):
        #    yield move
        #    move = map(sum, zip(move, direction))

def get_col_char(col):
    """ Convert 1, 2, etc. to 'a', 'b', etc. """
    return chr(ord('a')+col)

def moves_string(moves):
    """ Return the given list of coordinates as a nicely formatted list of 
    moves. Example: [(2,3),(5,2)] -> 'c4, f3' """
    s = ""
    for i, move in enumerate(moves):
        if i == len(moves)-1:
            s += move_string(move)
        else:
            s += move_string(move) + ', '
    return s

def print_moves(moves):
    """ Print the list of coordinates. """
    print(moves_string(moves))

def move_string(move):
    """ Convert a numeric (x,y) coordinate like (2,3) into a piece name like 'c4'. """
    (x,y) = move
    return get_col_char(x)+str(y+1)

