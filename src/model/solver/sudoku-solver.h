#include <set>

#include "sudoku-board.h"

namespace sudoku {

class SudokuSolver {

    public:
        static bool solve(SudokuBoard *unsolved, SudokuBoard *solved, BoardCell from = BoardCell(0, 0));

        static bool isValidGuess(SudokuBoard *board, BoardCell cell, char value);

        static bool isSolved(SudokuBoard board);

};

}