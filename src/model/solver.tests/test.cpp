#define CATCH_CONFIG_MAIN
#include <stdio.h>
#include "catch.h"

const int _ = -1;

/*
export default function solve(board: IBoard, cell: [number, number] = [0, 0]): boolean {
    let emptyCell = board.findFirstEmptyCell(cell);
    if (!emptyCell)
        // solved, or impossible to solve
        return true;

    for (let guess = 1; guess <= 9; guess++) {
        if (board.isValidGuess(emptyCell, guess)) {
            board.setCell(emptyCell, guess);
            if (solve(board, emptyCell))
                return true;
            else
                board.clearCell(emptyCell);
        }
    }

    return false;
}
*/

struct BoardSize {
    int rows;
    int cols;

    BoardSize(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
    }
};

struct BoardCell {
    char row;
    char col;

    BoardCell(int row, int col) {
        this->row = row;
        this->col = col;
    }
};

class SudokuBoard {

    char BOARD_SIZE = 9;
    std::vector<std::vector<char>> board;

    public:
        SudokuBoard(std::vector<std::vector<char>> matrix) {
            board = matrix;
        }
        SudokuBoard() {
            board.resize(BOARD_SIZE);
            for (int i = 0; i < BOARD_SIZE; i++)
                board[i].resize(BOARD_SIZE);
        }

        static SudokuBoard empty() {
            return SudokuBoard();
        }

        bool findFirstEmptyCell(BoardCell *emptyCell) {
            for (int row = 0; row < BOARD_SIZE; row += 1) {
                for (int col = 0; col < BOARD_SIZE; col += 1) {
                    if (board[row][col] == _) {
                        *emptyCell = BoardCell(row, col);
                        return true;
                    }
                }
            }

            return false;
        }

        void setValue(BoardCell cell, char value) {
            board[cell.row][cell.col] = value;
        }

        char getValue(BoardCell cell) {
            return board[cell.row][cell.col];
        }

        void clearValue(BoardCell cell) {
            board[cell.row][cell.col] = _;
        }

        BoardSize size() {
            return BoardSize(
                board.size(),
                board[0].size()
            );
        }
};


class SudokuSolver {

    public:
        static bool solve(SudokuBoard *unsolved, SudokuBoard *solved, BoardCell from = BoardCell(0, 0)) {

            BoardCell emptyCell = BoardCell(_, _);
            bool foundEmptyCell = unsolved->findFirstEmptyCell(&emptyCell);
            if (!foundEmptyCell)
                // solved, or impossible to solve
                return true;

            for (int guess = 1; guess <= 9; guess++) {
                if (SudokuSolver::isValidGuess(unsolved, emptyCell, guess)) {
                    unsolved->setValue(emptyCell, guess);

                    if (solve(unsolved, solved, emptyCell)) {
                        *solved = *unsolved;
                        return true;
                    } else {
                        unsolved->clearValue(emptyCell);
                    }
                }
            }

            return false;

        }


        static bool isValidGuess(SudokuBoard *board, BoardCell cell, char value) {
            const auto size = board->size();

            for (int i = 0; i < size.rows; i++) {
                if ( value == board->getValue(BoardCell(cell.row, i)) || value == board->getValue(BoardCell(i, cell.col)) ) {
                    return false;
                }
            }

            const std::vector<int> floorMultiplesOfThree = { 0, 0 , 0, 3, 3, 3, 6, 6, 6};
            auto startRow = floorMultiplesOfThree[cell.row];
            auto startCol = floorMultiplesOfThree[cell.col];

            for (int row = 0; row < 3; row += 1) {
                for (int col = 0; col < 3; col += 1) {
                    const auto cellValue = board->getValue(
                        BoardCell(startRow + row,  startCol + col)
                    );

                    if (cellValue == value) {
                        return false;
                    }
                }
            }

            return true;
        }

        static bool isSolved(SudokuBoard board) {
            std::set<char> rows;
            std::set<char> cols;
            std::set<char> squares;

            std::vector<char> floorMultipleOf3 { 0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9 };

            for (int i = 0; i < board.size().rows; i++) {
                // add every value to the set of row values
                for (int j = 0; j < board.size().cols; j++) {
                    auto rowValue = board.getValue(BoardCell(i, j));
                    rows.insert(rowValue);

                    auto colValue = board.getValue(BoardCell(j, i));
                    cols.insert(colValue);
                }

                // every value should be presented once in the set of row values
                if(rows.size() + cols.size() != board.size().rows * 2) {
                    // number not in set
                    return false;
                }
            }
            for (int rowStep = 0; rowStep < 9; rowStep += 3) {
                for (int colStep = 0; colStep < 9; colStep += 3) {
                    for (int rowI = 0; rowI < 3; rowI += 1) {
                        for (int colI = 0; colI < 3; colI += 1) {
                            squares.insert(board.getValue(BoardCell(rowStep + rowI, colStep + colI)));
                        }
                    }
                    if(squares.size() != cols.size()) {
                        // to few squares in set
                        return false;
                    }
                    squares.clear();
                }
            }

        return true;
    }

};

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

    for (int i = 0; i < solved.size().rows; i++) {
    // add every value to the set of row values
        for (int j = 0; j < solved.size().cols; j++) {
            int value = solved.getValue(BoardCell(i, j));
            std::cout << value << " ";
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