#define CATCH_CONFIG_MAIN
#include <stdio.h>
#include "catch.h"

#include "../sudoku-board.h"
#include "../sudoku-solver.h"

namespace sudoku {

TEST_CASE("Can solve one missing", "[solve]" ) {
    std::vector<std::vector<char>> notSolved = {
        {1, 2, 3,  4, 5, 6,  7, 8, 9},
        {4, 5, 6,  7, 8, 9,  1, 2, 3},
        {7, 8, 9,  1, 2, 3,  4, 5, 6},

        {2, 1, 4,  3, 6, 5,  8, 9, 7},
        {3, 6, 5,  8, 9, 7,  2, 1, 4},
        {8, 9, 7,  2, 1, 4,  3, 6, 5},

        {5, 3, 1,  6, 4, 2,  9, 7, 8},
        {6, 4, 2,  9, 7, 8,  5, 3, 1},
        {9, 7, 8,  5, 3, 1,  6, 4, _}
    };

    SudokuBoard board = notSolved;

    SudokuBoard solved;
    auto isSolved = SudokuSolver::solve(&board, &solved);

    REQUIRE (isSolved == true);
    REQUIRE (SudokuSolver::isSolved(solved) == true);
}

TEST_CASE("Can solve actual sudoku", "[solve]" ) {
    std::vector<std::vector<char>> notSolved = {
        { 5, 3, _,   _, 7, _,   _, _, _ },
        { 6, _, _,   1, 9, 5,   _, _, _ },
        { _, 9, 8,   _, _, _,   _, 6, _ },

        { 8, _, _,   _, 6, _,   _, _, 3 },
        { 4, _, _,   8, _, 3,   _, _, 1 },
        { 7, _, _,   _, 2, _,   _, _, 6 },

        { _, 6, _,   _, _, _,   2, 8, _ },
        { _, _, _,   4, 1, 9,   _, _, 5 },
        { _, _, _,   _, 8, _,   _, 7, 9 }
    };

    SudokuBoard board = notSolved;

    SudokuBoard solved = SudokuBoard::empty();
    auto isSolved = SudokuSolver::solve(&board, &solved);

    for (auto row: solved) {
        for (auto col: row) {
            std::cout << std::to_string(col) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << solved.getValue(BoardCell(0, 2)) << " ";
    REQUIRE (isSolved == true);
//    REQUIRE (solved.getValue(BoardCell(0, 2)) == 5);

    REQUIRE (SudokuSolver::isSolved(solved) == true);
}

////////// CHECK SOLVE

TEST_CASE("Can check solved sudoku", "[solve]" ) {
    std::vector<std::vector<char>> notSolved = {
        {1, 2, 3,  4, 5, 6,  7, 8, 9},
        {4, 5, 6,  7, 8, 9,  1, 2, 3},
        {7, 8, 9,  1, 2, 3,  4, 5, 6},

        {2, 1, 4,  3, 6, 5,  8, 9, 7},
        {3, 6, 5,  8, 9, 7,  2, 1, 4},
        {8, 9, 7,  2, 1, 4,  3, 6, 5},

        {5, 3, 1,  6, 4, 2,  9, 7, 8},
        {6, 4, 2,  9, 7, 8,  5, 3, 1},
        {9, 7, 8,  5, 3, 1,  6, 4, 2}
    };

    SudokuBoard board = notSolved;

    SudokuBoard solved;
    auto isSolved = SudokuSolver::solve(&board, &solved);

    REQUIRE (isSolved == true);
}

TEST_CASE("With duplicate rows are not solved", "[solve]" ) {
    std::vector<std::vector<char>> notSolved = {
        {1, 2, 3,  4, 5, 6,  7, 8, 9},
        {1, 2, 3,  4, 5, 6,  7, 8, 9},
        {1, 2, 3,  4, 5, 6,  7, 8, 9},

        {1, 2, 3,  4, 5, 6,  7, 8, 9},
        {1, 2, 3,  4, 5, 6,  7, 8, 9},
        {1, 2, 3,  4, 5, 6,  7, 8, 9},

        {1, 2, 3,  4, 5, 6,  7, 8, 9},
        {1, 2, 3,  4, 5, 6,  7, 8, 9},
        {1, 2, 3,  4, 5, 6,  7, 8, 9},
    };

    auto solved = SudokuSolver::isSolved(notSolved);

    REQUIRE (solved == false);
}

TEST_CASE("With duplicate squares are not solved", "[solve]" ) {
    std::vector<std::vector<char>> notSolved = {
        {1, 2, 3,  4, 5, 6,  7, 8, 9},
        {9, 1, 2,  3, 4, 5,  6, 7, 8},
        {8, 9, 1,  2, 3, 4,  5, 6, 7},

        {7, 8, 9,  1, 2, 3,  4, 5, 6},
        {6, 7, 8,  9, 1, 2,  3, 4, 5},
        {5, 6, 7,  8, 9, 1,  2, 3, 4},

        {4, 5, 6,  7, 8, 9,  1, 2, 3},
        {3, 4, 5,  6, 7, 8,  9, 1, 2},
        {2, 3, 4,  5, 6, 7,  8, 9, 1},
    };

    auto solved = SudokuSolver::isSolved(notSolved);

    REQUIRE (solved == false);
}

TEST_CASE("Find first empty board cell", "[firstEmpty]" ) {
    auto board = SudokuBoard::empty();

    BoardCell emptyCell = BoardCell(0, 0);
    board.findFirstEmptyCell(&emptyCell);

    REQUIRE( emptyCell.row == 0 );
    REQUIRE( emptyCell.col == 0 );
}

TEST_CASE("Set cell", "[setCell]" ) {
    auto board = SudokuBoard::empty();

    board.setValue(BoardCell(0, 0), 9);
    auto testValue = board.getValue(BoardCell(0, 0));

    REQUIRE( testValue == 9 );
}

}
