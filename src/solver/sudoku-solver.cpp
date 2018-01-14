#ifndef SUDOKU_SOLVER
#define SUDOKU_SOLVER

#include "sudoku-solver.h"
#include "sudoku-board.h"

namespace sudoku {

bool SudokuSolver::solve(SudokuBoard *unsolved, SudokuBoard *solved, BoardCell from) {

    bool foundEmptyCell = unsolved->findFirstEmptyCell(&from);
    if (!foundEmptyCell)
        return SudokuSolver::isSolved(*unsolved);

    for (int guess = 1; guess <= 9; guess++) {
        if (SudokuSolver::isValidGuess(unsolved, from, guess)) {
            unsolved->setValue(from, guess);

            if (solve(unsolved, solved, from)) {
                *solved = *unsolved;
                return true;
            } else {
                unsolved->clearValue(from);
            }
        }
    }

    return false;

}

bool SudokuSolver::isValidGuess(SudokuBoard *board, BoardCell cell, char value) {
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

bool SudokuSolver::isSolved(SudokuBoard board) {
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

}

#endif