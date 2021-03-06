# Sudoku IRL

![alt text](logo.png "Sudoku unicorn")

> Using computer vision and machine learning to solve a sudodu from images with OpenCV c++

## Challenges

- Finding the game square
- Straightening the game
- Recognizing digits, both machine- and handwritten

## Assumptions

- The puzzle is the largest element in the picture

## Steps

* ~~Segment the picture with thresholding~~
* ~~Find the game board outer lines~~
* ~~Find the game board corners to adjust skew~~
* ~~Straighten the game to a new image of only the game~~
* ~~Divide puzzle into 9*9 grid by approximation~~
* ~~Train SVM to understand handwritten digits~~~
* Train SVM to understand digital digits
* ~~Parse the numbers in the grid~~
* ~~Predict digits by SVM~~
* Solve the sudoku

## Maybe
~~Erode away noise before predictions~~
Debug flags
## Build

Prerequisites
* OpenCV
* cmake

```
$ cmake .
```

```
$ make
```

```
$ ./sudoku ./complex.jpg
```