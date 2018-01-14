#ifndef SUDOKU_BOARD
#define SUDOKU_BOARD

#include <stdio.h>
#include <vector>

const int _ = -1;

namespace sudoku {

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

        bool findFirstEmptyCell(BoardCell *emptyCell);

        void setValue(BoardCell cell, char value);

        char getValue(BoardCell cell);

        void clearValue(BoardCell cell);

        BoardSize size();

        // Delegate iterator to vector
        std::vector<std::vector<char>>::iterator begin();

        std::vector<std::vector<char>>::iterator end();
};

}

#endif
