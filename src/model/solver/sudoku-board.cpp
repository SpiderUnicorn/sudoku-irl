#include "./sudoku-board.h"

#include <stdio.h>
#include <vector>

namespace sudoku {

bool SudokuBoard::findFirstEmptyCell(BoardCell *emptyCell) {
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

void SudokuBoard::setValue(BoardCell cell, char value) {
    board[cell.row][cell.col] = value;
}

char SudokuBoard::getValue(BoardCell cell) {
    return board[cell.row][cell.col];
}

void SudokuBoard::clearValue(BoardCell cell) {
    board[cell.row][cell.col] = _;
}

BoardSize SudokuBoard::size() {
    return BoardSize(
        board.size(),
        board[0].size()
    );
}

// Delegate iterator to vector
std::vector<std::vector<char>>::iterator SudokuBoard::begin() {
    return board.begin();
}

std::vector<std::vector<char>>::iterator SudokuBoard::end() {
    return board.end();
}

}