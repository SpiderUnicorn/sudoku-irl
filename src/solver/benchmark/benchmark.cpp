#include <benchmark/benchmark.h>
#include "../sudoku-board.h"
#include "../sudoku-solver.h"

static void Complicated_sudoku(benchmark::State& state) {
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

    sudoku::SudokuBoard board = notSolved;

    sudoku::SudokuBoard solved;
    for (auto _ : state)
      auto isSolved = sudoku::SudokuSolver::solve(&board, &solved);
}

BENCHMARK(Complicated_sudoku);

BENCHMARK_MAIN();