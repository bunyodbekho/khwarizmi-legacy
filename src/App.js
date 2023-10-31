import { useState } from "react";
// import data from "./Data.json";

const data = [
  {
    id: "0",
    type: "sorting",
    name: "bread sorting",
    python: `
    def bead_sort(sequence: list) -> list:
    """
    >>> bead_sort([6, 11, 12, 4, 1, 5])
    [1, 4, 5, 6, 11, 12]

    >>> bead_sort([9, 8, 7, 6, 5, 4 ,3, 2, 1])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    >>> bead_sort([5, 0, 4, 3])
    [0, 3, 4, 5]

    >>> bead_sort([8, 2, 1])
    [1, 2, 8]

    >>> bead_sort([1, .9, 0.0, 0, -1, -.9])
    Traceback (most recent call last):
        ...
    TypeError: Sequence must be list of non-negative integers

    >>> bead_sort("Hello world")
    Traceback (most recent call last):
        ...
    TypeError: Sequence must be list of non-negative integers
    """
    if any(not isinstance(x, int) or x < 0 for x in sequence):
        raise TypeError("Sequence must be list of non-negative integers")
    for _ in range(len(sequence)):
        for i, (rod_upper, rod_lower) in enumerate(zip(sequence, sequence[1:])):
            if rod_upper > rod_lower:
                sequence[i] -= rod_upper - rod_lower
                sequence[i + 1] += rod_upper - rod_lower
    return sequence


    if __name__ == "__main__":
        assert bead_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
        assert bead_sort([7, 9, 4, 3, 5]) == [3, 4, 5, 7, 9]
        `,
    js: `
        export function beadSort (sequence) {
        /* Let's ensure our sequence has only Positive Integers */
        if (sequence.some((integer) => integer < 0)) {
            throw RangeError('Sequence must be a list of Positive integers Only!')
        }
        
        const sequenceLength = sequence.length
        const max = Math.max(...sequence)
        
        // Set initial Grid
        const grid = sequence.map(number => {
            const maxArr = new Array(max)
        
            for (let i = 0; i < number; i++) {
            maxArr[i] = '*'
            }
        
            return maxArr
        })
        
        // Drop the Beads!
        for (let col = 0; col < max; col++) {
            let beadsCount = 0
        
            for (let row = 0; row < sequenceLength; row++) {
            if (grid[row][col] === '*') {
                beadsCount++
            }
            }
        
            for (let row = sequenceLength - 1; row > -1; row--) {
            if (beadsCount) {
                grid[row][col] = '*'
                beadsCount--
            } else {
                grid[row][col] = undefined
            }
            }
        }
        
        /* Finally, let's turn our Bead rows into their Respective Numbers */
        return grid.map((beadArray) => {
            const beadsArray = beadArray.filter(bead => bead === '*')
            return beadsArray.length
        })
        }
    `,
    java: `
    package com.thealgorithms.sorts;

    // BeadSort Algorithm(wikipedia) : https://en.wikipedia.org/wiki/Bead_sort
    // BeadSort can't sort negative number, Character, String. It can sort positive number only

    public class BeadSort {
        public int[] sort(int[] unsorted) {
            int[] sorted = new int[unsorted.length];
            int max = 0;
            for (int i = 0; i < unsorted.length; i++) {
                max = Math.max(max, unsorted[i]);
            }

            char[][] grid = new char[unsorted.length][max];
            int[] count = new int[max];

            for (int i = 0; i < unsorted.length; i++) {
                for (int j = 0; j < max; j++) {
                    grid[i][j] = '-';
                }
            }

            for (int i = 0; i < max; i++) {
                count[i] = 0;
            }

            for (int i = 0; i < unsorted.length; i++) {
                int k = 0;
                for (int j = 0; j < unsorted[i]; j++) {
                    grid[count[max - k - 1]++][k] = '*';
                    k++;
                }
            }

            for (int i = 0; i < unsorted.length; i++) {
                int k = 0;
                for (int j = 0; j < max && grid[unsorted.length - 1 - i][j] == '*'; j++) {
                    k++;
                }
                sorted[i] = k;
            }
            return sorted;
        }
    }
    `,
  },
  {
    id: "1",
    type: "sorting",
    name: "binary insertion sort",
    python: `
    """
    This is a pure Python implementation of the binary insertion sort algorithm

    For doctests run following command:
    python -m doctest -v binary_insertion_sort.py
    or
    python3 -m doctest -v binary_insertion_sort.py

    For manual testing run:
    python binary_insertion_sort.py
    """


    def binary_insertion_sort(collection: list) -> list:
        """Pure implementation of the binary insertion sort algorithm in Python
        :param collection: some mutable ordered collection with heterogeneous
        comparable items inside
        :return: the same collection ordered by ascending

        Examples:
        >>> binary_insertion_sort([0, 4, 1234, 4, 1])
        [0, 1, 4, 4, 1234]
        >>> binary_insertion_sort([]) == sorted([])
        True
        >>> binary_insertion_sort([-1, -2, -3]) == sorted([-1, -2, -3])
        True
        >>> lst = ['d', 'a', 'b', 'e', 'c']
        >>> binary_insertion_sort(lst) == sorted(lst)
        True
        >>> import random
        >>> collection = random.sample(range(-50, 50), 100)
        >>> binary_insertion_sort(collection) == sorted(collection)
        True
        >>> import string
        >>> collection = random.choices(string.ascii_letters + string.digits, k=100)
        >>> binary_insertion_sort(collection) == sorted(collection)
        True
        """

        n = len(collection)
        for i in range(1, n):
            val = collection[i]
            low = 0
            high = i - 1

            while low <= high:
                mid = (low + high) // 2
                if val < collection[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            for j in range(i, low, -1):
                collection[j] = collection[j - 1]
            collection[low] = val
        return collection


        if __name__ == "__main__":
            user_input = input("Enter numbers separated by a comma:\n").strip()
            unsorted = [int(item) for item in user_input.split(",")]
            print(binary_insertion_sort(unsorted))
            `,
    js: `
        /**
         * Pure Implementation of Binary Search Algorithm
         *
         * Binary insertion sort is a sorting algorithm similar to insertion sort,
         * but instead of using linear search to find the position
         * where the element should be inserted, we use binary search.
         * Thus, we reduce the number of comparisons for inserting one element from O(N)
         * (Time complexity in Insertion Sort) to O(log N).
         *
         */

        /**
         * Search the key element in the array from start position to end position.
         *
         * @param {Array} array Array of numbers.
         * @param {Number} key Value to be searched
         * @param {Number} start start index position of array
         * @param {Number} end end index position of array
         * @return {Number} Position of the key element
         */
        function binarySearch (array, key, start, end) {
        if (start === end) {
            if (array[start] > key) {
            return start
            } else {
            return start + 1
            }
        }

        if (start > end) {
            return start
        }

        const mid = Math.floor((start + end) / 2)

        if (array[mid] < key) {
            return binarySearch(array, key, mid + 1, end)
        } else if (array[mid] > key) {
            return binarySearch(array, key, start, mid - 1)
        } else {
            return mid
        }
        }

        /**
         * Binary Insertion Sort
         *
         * @param {Array} list List to be sorted.
         * @return {Array} The sorted list.
         */
        export function binaryInsertionSort (array) {
        const totalLength = array.length
        for (let i = 1; i < totalLength; i += 1) {
            const key = array[i]
            const indexPosition = binarySearch(array, key, 0, i - 1)
            array.splice(i, 1)
            array.splice(indexPosition, 0, key)
        }
        return array
        }
    `,
    java: `
    package com.thealgorithms.sorts;

    public class BinaryInsertionSort {

    // Binary Insertion Sort method
    public int[] binaryInsertSort(int[] array) {
        for (int i = 1; i < array.length; i++) {
            int temp = array[i];
            int low = 0;
            int high = i - 1;

            while (low <= high) {
                int mid = (low + high) / 2;
                if (temp < array[mid]) {
                    high = mid - 1;
                } else {
                    low = mid + 1;
                }
            }

            for (int j = i; j >= low + 1; j--) {
                array[j] = array[j - 1];
            }

            array[low] = temp;
        }
        return array;
    }
}
    `,
  },
  {
    id: "2",
    type: "sorting",
    name: "bubble sort",
    python: `
    from typing import Any


    def bubble_sort(collection: list[Any]) -> list[Any]:
    """Pure implementation of bubble sort algorithm in Python

    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending

    Examples:
    >>> bubble_sort([0, 5, 2, 3, 2])
    [0, 2, 2, 3, 5]
    >>> bubble_sort([0, 5, 2, 3, 2]) == sorted([0, 5, 2, 3, 2])
    True
    >>> bubble_sort([]) == sorted([])
    True
    >>> bubble_sort([-2, -45, -5]) == sorted([-2, -45, -5])
    True
    >>> bubble_sort([-23, 0, 6, -4, 34]) == sorted([-23, 0, 6, -4, 34])
    True
    >>> bubble_sort(['d', 'a', 'b', 'e', 'c']) == sorted(['d', 'a', 'b', 'e', 'c'])
    True
    >>> import random
    >>> collection = random.sample(range(-50, 50), 100)
    >>> bubble_sort(collection) == sorted(collection)
    True
    >>> import string
    >>> collection = random.choices(string.ascii_letters + string.digits, k=100)
    >>> bubble_sort(collection) == sorted(collection)
    True
    """
    length = len(collection)
    for i in reversed(range(length)):
        swapped = False
        for j in range(i):
            if collection[j] > collection[j + 1]:
                swapped = True
                collection[j], collection[j + 1] = collection[j + 1], collection[j]
        if not swapped:
            break  # Stop iteration if the collection is sorted.
    return collection


    if __name__ == "__main__":
        import doctest
        import time

        doctest.testmod()

        user_input = input("Enter numbers separated by a comma:").strip()
        unsorted = [int(item) for item in user_input.split(",")]
        start = time.process_time()
        print(*bubble_sort(unsorted), sep=",")
        print(f"Processing time: {(time.process_time() - start)%1e9 + 7}")
    `,
    js: `
    /* Bubble Sort is an algorithm to sort an array. It
    *  compares adjacent element and swaps their position
    *  The big O on bubble sort in worst and best case is O(N^2).
    *  Not efficient.
    *  Somehow if the array is sorted or nearly sorted then we can optimize bubble sort by adding a flag.
    *
    *  In bubble sort, we keep iterating while something was swapped in
    *  the previous inner-loop iteration. By swapped I mean, in the
    *  inner loop iteration, we check each number if the number proceeding
    *  it is greater than itself, if so we swap them.
    *
    *  Wikipedia: https://en.wikipedia.org/wiki/Bubble_sort
    *  Animated Visual: https://www.toptal.com/developers/sorting-algorithms/bubble-sort
    */

    /**
     * Using 2 for loops.
     */
    export function bubbleSort (items) {
    const length = items.length
    let noSwaps

    for (let i = length; i > 0; i--) {
        // flag for optimization
        noSwaps = true
        // Number of passes
        for (let j = 0; j < (i - 1); j++) {
        // Compare the adjacent positions
        if (items[j] > items[j + 1]) {
            // Swap the numbers
            [items[j], items[j + 1]] = [items[j + 1], items[j]]
            noSwaps = false
        }
        }
        if (noSwaps) {
        break
        }
    }

    return items
    }

    /**
     * Using a while loop and a for loop.
     */
    export function alternativeBubbleSort (arr) {
    let swapped = true

    while (swapped) {
        swapped = false
        for (let i = 0; i < arr.length - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            [arr[i], arr[i + 1]] = [arr[i + 1], arr[i]]
            swapped = true
        }
        }
    }

    return arr
    }
    `,
    java: `
    package com.thealgorithms.sorts;

import static com.thealgorithms.sorts.SortUtils.*;

/**
 * @author Varun Upadhyay (https://github.com/varunu28)
 * @author Podshivalov Nikita (https://github.com/nikitap492)
 * @see SortAlgorithm
 */
class BubbleSort implements SortAlgorithm {

    /**
     * Implements generic bubble sort algorithm.
     *
     * @param array the array to be sorted.
     * @param <T> the type of elements in the array.
     * @return the sorted array.
     */
    @Override
    public <T extends Comparable<T>> T[] sort(T[] array) {
        for (int i = 1, size = array.length; i < size; ++i) {
            boolean swapped = false;
            for (int j = 0; j < size - i; ++j) {
                if (greater(array[j], array[j + 1])) {
                    swap(array, j, j + 1);
                    swapped = true;
                }
            }
            if (!swapped) {
                break;
            }
        }
        return array;
    }
}
    `,
  },
  {
    id: "3",
    type: "sorting",
    name: "bogo sort",
    python: `
    """
This is a pure Python implementation of the bogosort algorithm,
also known as permutation sort, stupid sort, slowsort, shotgun sort, or monkey sort.
Bogosort generates random permutations until it guesses the correct one.

More info on: https://en.wikipedia.org/wiki/Bogosort

For doctests run following command:
python -m doctest -v bogo_sort.py
or
python3 -m doctest -v bogo_sort.py
For manual testing run:
python bogo_sort.py
"""

import random


def bogo_sort(collection):
    """Pure implementation of the bogosort algorithm in Python
    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending
    Examples:
    >>> bogo_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> bogo_sort([])
    []
    >>> bogo_sort([-2, -5, -45])
    [-45, -5, -2]
    """

    def is_sorted(collection):
        for i in range(len(collection) - 1):
            if collection[i] > collection[i + 1]:
                return False
        return True

    while not is_sorted(collection):
        random.shuffle(collection)
    return collection


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(bogo_sort(unsorted))
    `,
    js: `
    /**
 * Checks whether the given array is sorted in ascending order.
 */
export function isSorted (array) {
  const length = array.length
  for (let i = 0; i < length - 1; i++) {
    if (array[i] > array[i + 1]) {
      return false
    }
  }
  return true
}

/**
 * Shuffles the given array randomly in place.
 */
function shuffle (array) {
  for (let i = array.length - 1; i; i--) {
    const m = Math.floor(Math.random() * i)
    const n = array[i - 1]
    array[i - 1] = array[m]
    array[m] = n
  }
}

/**
 * Implementation of the bogosort algorithm.
 *
 * This sorting algorithm randomly rearranges the array until it is sorted.
 *
 * For more information see: https://en.wikipedia.org/wiki/Bogosort
 */
export function bogoSort (items) {
  while (!isSorted(items)) {
    shuffle(items)
  }
  return items
}
    `,
    java: `
    package com.thealgorithms.sorts;

import java.util.Random;

/**
 * @author Podshivalov Nikita (https://github.com/nikitap492)
 * @see SortAlgorithm
 */
public class BogoSort implements SortAlgorithm {

    private static final Random random = new Random();

    private static <T extends Comparable<T>> boolean isSorted(T[] array) {
        for (int i = 0; i < array.length - 1; i++) {
            if (SortUtils.less(array[i + 1], array[i])) {
                return false;
            }
        }
        return true;
    }

    // Randomly shuffles the array
    private static <T> void nextPermutation(T[] array) {
        int length = array.length;

        for (int i = 0; i < array.length; i++) {
            int randomIndex = i + random.nextInt(length - i);
            SortUtils.swap(array, randomIndex, i);
        }
    }

    public <T extends Comparable<T>> T[] sort(T[] array) {
        while (!isSorted(array)) {
            nextPermutation(array);
        }
        return array;
    }

    // Driver Program
    public static void main(String[] args) {
        // Integer Input
        Integer[] integers = {4, 23, 6, 78, 1, 54, 231, 9, 12};

        BogoSort bogoSort = new BogoSort();

        // print a sorted array
        SortUtils.print(bogoSort.sort(integers));

        // String Input
        String[] strings = {"c", "a", "e", "b", "d"};

        SortUtils.print(bogoSort.sort(strings));
    }
}
    `,
  },
  {
    id: "4",
    type: "quest",
    name: "binary search",
    python: `
    """
This is pure Python implementation of binary search algorithms

For doctests run following command:
python3 -m doctest -v binary_search.py

For manual testing run:
python3 binary_search.py
"""
from __future__ import annotations

import bisect


def bisect_left(
    sorted_collection: list[int], item: int, lo: int = 0, hi: int = -1
) -> int:
    """
    Locates the first element in a sorted array that is larger or equal to a given
    value.

    It has the same interface as
    https://docs.python.org/3/library/bisect.html#bisect.bisect_left .

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item to bisect
    :param lo: lowest index to consider (as in sorted_collection[lo:hi])
    :param hi: past the highest index to consider (as in sorted_collection[lo:hi])
    :return: index i such that all values in sorted_collection[lo:i] are < item and all
        values in sorted_collection[i:hi] are >= item.

    Examples:
    >>> bisect_left([0, 5, 7, 10, 15], 0)
    0

    >>> bisect_left([0, 5, 7, 10, 15], 6)
    2

    >>> bisect_left([0, 5, 7, 10, 15], 20)
    5

    >>> bisect_left([0, 5, 7, 10, 15], 15, 1, 3)
    3

    >>> bisect_left([0, 5, 7, 10, 15], 6, 2)
    2
    """
    if hi < 0:
        hi = len(sorted_collection)

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if sorted_collection[mid] < item:
            lo = mid + 1
        else:
            hi = mid

    return lo


def bisect_right(
    sorted_collection: list[int], item: int, lo: int = 0, hi: int = -1
) -> int:
    """
    Locates the first element in a sorted array that is larger than a given value.

    It has the same interface as
    https://docs.python.org/3/library/bisect.html#bisect.bisect_right .

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item to bisect
    :param lo: lowest index to consider (as in sorted_collection[lo:hi])
    :param hi: past the highest index to consider (as in sorted_collection[lo:hi])
    :return: index i such that all values in sorted_collection[lo:i] are <= item and
        all values in sorted_collection[i:hi] are > item.

    Examples:
    >>> bisect_right([0, 5, 7, 10, 15], 0)
    1

    >>> bisect_right([0, 5, 7, 10, 15], 15)
    5

    >>> bisect_right([0, 5, 7, 10, 15], 6)
    2

    >>> bisect_right([0, 5, 7, 10, 15], 15, 1, 3)
    3

    >>> bisect_right([0, 5, 7, 10, 15], 6, 2)
    2
    """
    if hi < 0:
        hi = len(sorted_collection)

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if sorted_collection[mid] <= item:
            lo = mid + 1
        else:
            hi = mid

    return lo


def insort_left(
    sorted_collection: list[int], item: int, lo: int = 0, hi: int = -1
) -> None:
    """
    Inserts a given value into a sorted array before other values with the same value.

    It has the same interface as
    https://docs.python.org/3/library/bisect.html#bisect.insort_left .

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item to insert
    :param lo: lowest index to consider (as in sorted_collection[lo:hi])
    :param hi: past the highest index to consider (as in sorted_collection[lo:hi])

    Examples:
    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_left(sorted_collection, 6)
    >>> sorted_collection
    [0, 5, 6, 7, 10, 15]

    >>> sorted_collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    >>> item = (5, 5)
    >>> insort_left(sorted_collection, item)
    >>> sorted_collection
    [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    >>> item is sorted_collection[1]
    True
    >>> item is sorted_collection[2]
    False

    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_left(sorted_collection, 20)
    >>> sorted_collection
    [0, 5, 7, 10, 15, 20]

    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_left(sorted_collection, 15, 1, 3)
    >>> sorted_collection
    [0, 5, 7, 15, 10, 15]
    """
    sorted_collection.insert(bisect_left(sorted_collection, item, lo, hi), item)


def insort_right(
    sorted_collection: list[int], item: int, lo: int = 0, hi: int = -1
) -> None:
    """
    Inserts a given value into a sorted array after other values with the same value.

    It has the same interface as
    https://docs.python.org/3/library/bisect.html#bisect.insort_right .

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item to insert
    :param lo: lowest index to consider (as in sorted_collection[lo:hi])
    :param hi: past the highest index to consider (as in sorted_collection[lo:hi])

    Examples:
    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_right(sorted_collection, 6)
    >>> sorted_collection
    [0, 5, 6, 7, 10, 15]

    >>> sorted_collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    >>> item = (5, 5)
    >>> insort_right(sorted_collection, item)
    >>> sorted_collection
    [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    >>> item is sorted_collection[1]
    False
    >>> item is sorted_collection[2]
    True

    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_right(sorted_collection, 20)
    >>> sorted_collection
    [0, 5, 7, 10, 15, 20]

    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_right(sorted_collection, 15, 1, 3)
    >>> sorted_collection
    [0, 5, 7, 15, 10, 15]
    """
    sorted_collection.insert(bisect_right(sorted_collection, item, lo, hi), item)


def binary_search(sorted_collection: list[int], item: int) -> int | None:
    """Pure implementation of binary search algorithm in Python

    Be careful collection must be ascending sorted, otherwise result will be
    unpredictable

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if item is not found

    Examples:
    >>> binary_search([0, 5, 7, 10, 15], 0)
    0

    >>> binary_search([0, 5, 7, 10, 15], 15)
    4

    >>> binary_search([0, 5, 7, 10, 15], 5)
    1

    >>> binary_search([0, 5, 7, 10, 15], 6)

    """
    left = 0
    right = len(sorted_collection) - 1

    while left <= right:
        midpoint = left + (right - left) // 2
        current_item = sorted_collection[midpoint]
        if current_item == item:
            return midpoint
        elif item < current_item:
            right = midpoint - 1
        else:
            left = midpoint + 1
    return None


def binary_search_std_lib(sorted_collection: list[int], item: int) -> int | None:
    """Pure implementation of binary search algorithm in Python using stdlib

    Be careful collection must be ascending sorted, otherwise result will be
    unpredictable

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if item is not found

    Examples:
    >>> binary_search_std_lib([0, 5, 7, 10, 15], 0)
    0

    >>> binary_search_std_lib([0, 5, 7, 10, 15], 15)
    4

    >>> binary_search_std_lib([0, 5, 7, 10, 15], 5)
    1

    >>> binary_search_std_lib([0, 5, 7, 10, 15], 6)

    """
    index = bisect.bisect_left(sorted_collection, item)
    if index != len(sorted_collection) and sorted_collection[index] == item:
        return index
    return None


def binary_search_by_recursion(
    sorted_collection: list[int], item: int, left: int, right: int
) -> int | None:
    """Pure implementation of binary search algorithm in Python by recursion

    Be careful collection must be ascending sorted, otherwise result will be
    unpredictable
    First recursion should be started with left=0 and right=(len(sorted_collection)-1)

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if item is not found

    Examples:
    >>> binary_search_by_recursion([0, 5, 7, 10, 15], 0, 0, 4)
    0

    >>> binary_search_by_recursion([0, 5, 7, 10, 15], 15, 0, 4)
    4

    >>> binary_search_by_recursion([0, 5, 7, 10, 15], 5, 0, 4)
    1

    >>> binary_search_by_recursion([0, 5, 7, 10, 15], 6, 0, 4)

    """
    if right < left:
        return None

    midpoint = left + (right - left) // 2

    if sorted_collection[midpoint] == item:
        return midpoint
    elif sorted_collection[midpoint] > item:
        return binary_search_by_recursion(sorted_collection, item, left, midpoint - 1)
    else:
        return binary_search_by_recursion(sorted_collection, item, midpoint + 1, right)


if __name__ == "__main__":
    user_input = input("Enter numbers separated by comma:\n").strip()
    collection = sorted(int(item) for item in user_input.split(","))
    target = int(input("Enter a single number to be found in the list:\n"))
    result = binary_search(collection, target)
    if result is None:
        print(f"{target} was not found in {collection}.")
    else:
        print(f"{target} was found at position {result} in {collection}.")
    `,
    js: `
    function binarySearchRecursive (arr, x, low = 0, high = arr.length - 1) {
      const mid = Math.floor(low + (high - low) / 2)
    
      if (high >= low) {
        if (arr[mid] === x) {
          // item found => return its index
          return mid
        }
    
        if (x < arr[mid]) {
          // arr[mid] is an upper bound for x, so if x is in arr => low <= x < mid
          return binarySearchRecursive(arr, x, low, mid - 1)
        } else {
          // arr[mid] is a lower bound for x, so if x is in arr => mid < x <= high
          return binarySearchRecursive(arr, x, mid + 1, high)
        }
      } else {
        // if low > high => we have searched the whole array without finding the item
        return -1
      }
    }
    function binarySearchIterative (arr, x, low = 0, high = arr.length - 1) {
      while (high >= low) {
        const mid = Math.floor(low + (high - low) / 2)
    
        if (arr[mid] === x) {
          // item found => return its index
          return mid
        }
    
        if (x < arr[mid]) {
          // arr[mid] is an upper bound for x, so if x is in arr => low <= x < mid
          high = mid - 1
        } else {
          // arr[mid] is a lower bound for x, so if x is in arr => mid < x <= high
          low = mid + 1
        }
      }
      // if low > high => we have searched the whole array without finding the item
      return -1
    }
    
    export { binarySearchIterative, binarySearchRecursive }
    `,
    java: `
    package com.thealgorithms.searches;

import com.thealgorithms.devutils.searches.SearchAlgorithm;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * Binary search is one of the most popular algorithms The algorithm finds the
 * position of a target value within a sorted array
 *
 * <p>
 * Worst-case performance O(log n) Best-case performance O(1) Average
 * performance O(log n) Worst-case space complexity O(1)
 *
 * @author Varun Upadhyay (https://github.com/varunu28)
 * @author Podshivalov Nikita (https://github.com/nikitap492)
 * @see SearchAlgorithm
 * @see IterativeBinarySearch
 */
class BinarySearch implements SearchAlgorithm {

    /**
     * @param array is an array where the element should be found
     * @param key is an element which should be found
     * @param <T> is any comparable type
     * @return index of the element
     */
    @Override
    public <T extends Comparable<T>> int find(T[] array, T key) {
        return search(array, key, 0, array.length - 1);
    }

    /**
     * This method implements the Generic Binary Search
     *
     * @param array The array to make the binary search
     * @param key The number you are looking for
     * @param left The lower bound
     * @param right The upper bound
     * @return the location of the key
     */
    private <T extends Comparable<T>> int search(T[] array, T key, int left, int right) {
        if (right < left) {
            return -1; // this means that the key not found
        }
        // find median
        int median = (left + right) >>> 1;
        int comp = key.compareTo(array[median]);

        if (comp == 0) {
            return median;
        } else if (comp < 0) {
            return search(array, key, left, median - 1);
        } else {
            return search(array, key, median + 1, right);
        }
    }

    // Driver Program
    public static void main(String[] args) {
        // Just generate data
        Random r = ThreadLocalRandom.current();

        int size = 100;
        int maxElement = 100000;

        Integer[] integers = IntStream.generate(() -> r.nextInt(maxElement)).limit(size).sorted().boxed().toArray(Integer[] ::new);

        // The element that should be found
        int shouldBeFound = integers[r.nextInt(size - 1)];

        BinarySearch search = new BinarySearch();
        int atIndex = search.find(integers, shouldBeFound);

        System.out.printf("Should be found: %d. Found %d at index %d. An array length %d%n", shouldBeFound, integers[atIndex], atIndex, size);

        int toCheck = Arrays.binarySearch(integers, shouldBeFound);
        System.out.printf("Found by system method at an index: %d. Is equal: %b%n", toCheck, toCheck == atIndex);
    }
}
    `,
  },
  {
    id: "5",
    type: "quest",
    name: "binary tree traversal",
    python: `
    """
This is pure Python implementation of tree traversal algorithms
"""
from __future__ import annotations

import queue


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.right = None
        self.left = None


def build_tree() -> TreeNode:
    print("\n********Press N to stop entering at any point of time********\n")
    check = input("Enter the value of the root node: ").strip().lower()
    q: queue.Queue = queue.Queue()
    tree_node = TreeNode(int(check))
    q.put(tree_node)
    while not q.empty():
        node_found = q.get()
        msg = f"Enter the left node of {node_found.data}: "
        check = input(msg).strip().lower() or "n"
        if check == "n":
            return tree_node
        left_node = TreeNode(int(check))
        node_found.left = left_node
        q.put(left_node)
        msg = f"Enter the right node of {node_found.data}: "
        check = input(msg).strip().lower() or "n"
        if check == "n":
            return tree_node
        right_node = TreeNode(int(check))
        node_found.right = right_node
        q.put(right_node)
    raise


def pre_order(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> pre_order(root)
    1,2,4,5,3,6,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    print(node.data, end=",")
    pre_order(node.left)
    pre_order(node.right)


def in_order(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> in_order(root)
    4,2,5,1,6,3,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    in_order(node.left)
    print(node.data, end=",")
    in_order(node.right)


def post_order(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> post_order(root)
    4,5,2,6,7,3,1,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    post_order(node.left)
    post_order(node.right)
    print(node.data, end=",")


def level_order(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> level_order(root)
    1,2,3,4,5,6,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    q: queue.Queue = queue.Queue()
    q.put(node)
    while not q.empty():
        node_dequeued = q.get()
        print(node_dequeued.data, end=",")
        if node_dequeued.left:
            q.put(node_dequeued.left)
        if node_dequeued.right:
            q.put(node_dequeued.right)


def level_order_actual(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> level_order_actual(root)
    1,
    2,3,
    4,5,6,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    q: queue.Queue = queue.Queue()
    q.put(node)
    while not q.empty():
        list_ = []
        while not q.empty():
            node_dequeued = q.get()
            print(node_dequeued.data, end=",")
            if node_dequeued.left:
                list_.append(node_dequeued.left)
            if node_dequeued.right:
                list_.append(node_dequeued.right)
        print()
        for node in list_:
            q.put(node)


# iteration version
def pre_order_iter(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> pre_order_iter(root)
    1,2,4,5,3,6,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    stack: list[TreeNode] = []
    n = node
    while n or stack:
        while n:  # start from root node, find its left child
            print(n.data, end=",")
            stack.append(n)
            n = n.left
        # end of while means current node doesn't have left child
        n = stack.pop()
        # start to traverse its right child
        n = n.right


def in_order_iter(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> in_order_iter(root)
    4,2,5,1,6,3,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    stack: list[TreeNode] = []
    n = node
    while n or stack:
        while n:
            stack.append(n)
            n = n.left
        n = stack.pop()
        print(n.data, end=",")
        n = n.right


def post_order_iter(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> post_order_iter(root)
    4,5,2,6,7,3,1,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    stack1, stack2 = [], []
    n = node
    stack1.append(n)
    while stack1:  # to find the reversed order of post order, store it in stack2
        n = stack1.pop()
        if n.left:
            stack1.append(n.left)
        if n.right:
            stack1.append(n.right)
        stack2.append(n)
    while stack2:  # pop up from stack2 will be the post order
        print(stack2.pop().data, end=",")


def prompt(s: str = "", width=50, char="*") -> str:
    if not s:
        return "\n" + width * char
    left, extra = divmod(width - len(s) - 2, 2)
    return f"{left * char} {s} {(left + extra) * char}"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print(prompt("Binary Tree Traversals"))

    node: TreeNode = build_tree()
    print(prompt("Pre Order Traversal"))
    pre_order(node)
    print(prompt() + "\n")

    print(prompt("In Order Traversal"))
    in_order(node)
    print(prompt() + "\n")

    print(prompt("Post Order Traversal"))
    post_order(node)
    print(prompt() + "\n")

    print(prompt("Level Order Traversal"))
    level_order(node)
    print(prompt() + "\n")

    print(prompt("Actual Level Order Traversal"))
    level_order_actual(node)
    print("*" * 50 + "\n")

    print(prompt("Pre Order Traversal - Iteration Version"))
    pre_order_iter(node)
    print(prompt() + "\n")

    print(prompt("In Order Traversal - Iteration Version"))
    in_order_iter(node)
    print(prompt() + "\n")

    print(prompt("Post Order Traversal - Iteration Version"))
    post_order_iter(node)
    print(prompt())
    `,
    js: "js",
    java: "java",
  },
  {
    id: "6",
    type: "quest",
    name: "double linear search",
    python: `
    from __future__ import annotations


def double_linear_search(array: list[int], search_item: int) -> int:
    """
    Iterate through the array from both sides to find the index of search_item.

    :param array: the array to be searched
    :param search_item: the item to be searched
    :return the index of search_item, if search_item is in array, else -1

    Examples:
    >>> double_linear_search([1, 5, 5, 10], 1)
    0
    >>> double_linear_search([1, 5, 5, 10], 5)
    1
    >>> double_linear_search([1, 5, 5, 10], 100)
    -1
    >>> double_linear_search([1, 5, 5, 10], 10)
    3
    """
    # define the start and end index of the given array
    start_ind, end_ind = 0, len(array) - 1
    while start_ind <= end_ind:
        if array[start_ind] == search_item:
            return start_ind
        elif array[end_ind] == search_item:
            return end_ind
        else:
            start_ind += 1
            end_ind -= 1
    # returns -1 if search_item is not found in array
    return -1


if __name__ == "__main__":
    print(double_linear_search(list(range(100)), 40))
    `,
    js: "js",
    java: "java",
  },
  {
    id: "7",
    type: "quest",
    name: "fibonacci search",
    python: `
    """
This is pure Python implementation of fibonacci search.

Resources used:
https://en.wikipedia.org/wiki/Fibonacci_search_technique

For doctests run following command:
python3 -m doctest -v fibonacci_search.py

For manual testing run:
python3 fibonacci_search.py
"""
from functools import lru_cache


@lru_cache
def fibonacci(k: int) -> int:
    """Finds fibonacci number in index k.

    Parameters
    ----------
    k :
        Index of fibonacci.

    Returns
    -------
    int
        Fibonacci number in position k.

    >>> fibonacci(0)
    0
    >>> fibonacci(2)
    1
    >>> fibonacci(5)
    5
    >>> fibonacci(15)
    610
    >>> fibonacci('a')
    Traceback (most recent call last):
    TypeError: k must be an integer.
    >>> fibonacci(-5)
    Traceback (most recent call last):
    ValueError: k integer must be greater or equal to zero.
    """
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 0:
        raise ValueError("k integer must be greater or equal to zero.")
    if k == 0:
        return 0
    elif k == 1:
        return 1
    else:
        return fibonacci(k - 1) + fibonacci(k - 2)


def fibonacci_search(arr: list, val: int) -> int:
    """A pure Python implementation of a fibonacci search algorithm.

    Parameters
    ----------
    arr
        List of sorted elements.
    val
        Element to search in list.

    Returns
    -------
    int
        The index of the element in the array.
        -1 if the element is not found.

    >>> fibonacci_search([4, 5, 6, 7], 4)
    0
    >>> fibonacci_search([4, 5, 6, 7], -10)
    -1
    >>> fibonacci_search([-18, 2], -18)
    0
    >>> fibonacci_search([5], 5)
    0
    >>> fibonacci_search(['a', 'c', 'd'], 'c')
    1
    >>> fibonacci_search(['a', 'c', 'd'], 'f')
    -1
    >>> fibonacci_search([], 1)
    -1
    >>> fibonacci_search([.1, .4 , 7], .4)
    1
    >>> fibonacci_search([], 9)
    -1
    >>> fibonacci_search(list(range(100)), 63)
    63
    >>> fibonacci_search(list(range(100)), 99)
    99
    >>> fibonacci_search(list(range(-100, 100, 3)), -97)
    1
    >>> fibonacci_search(list(range(-100, 100, 3)), 0)
    -1
    >>> fibonacci_search(list(range(-100, 100, 5)), 0)
    20
    >>> fibonacci_search(list(range(-100, 100, 5)), 95)
    39
    """
    len_list = len(arr)
    # Find m such that F_m >= n where F_i is the i_th fibonacci number.
    i = 0
    while True:
        if fibonacci(i) >= len_list:
            fibb_k = i
            break
        i += 1
    offset = 0
    while fibb_k > 0:
        index_k = min(
            offset + fibonacci(fibb_k - 1), len_list - 1
        )  # Prevent out of range
        item_k_1 = arr[index_k]
        if item_k_1 == val:
            return index_k
        elif val < item_k_1:
            fibb_k -= 1
        elif val > item_k_1:
            offset += fibonacci(fibb_k - 1)
            fibb_k -= 2
    else:
        return -1


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    `,
    js: `
    /****************************************************************************
 * Fibonacci Search JavaScript Implementation
 * Author   Alhassan Atama Isiaka
 * Version v1.0.0
 * Copyright 2020
 * https://github.com/komputarist
 *
 * This implementation is based on Generalizing the Fibonacci search we
 * define the Fibonacci search of degree K. Like the Fibonacci search,
 * which it reduces to for K = 2, the Fibonacci search of degree K
 * involves only addition and subtraction.
 *  Capocelli R.M. (1991) A Generalization of the Fibonacci Search. In:
 * Bergum G.E., Philippou A.N., Horadam A.F. (eds) Applications of Fibonacci
 * Numbers. Springer, Dordrecht. https://doi.org/10.1007/978-94-011-3586-3_9
 *
 * This snippet is free. Feel free to improve on it
 *
 * We define a function fibonacciSearch() that takes an array of numbers,
 * the item (number) to be searched for and the length of the items in the array
 ****************************************************************************/

export const fibonacciSearch = (arr, x, n) => {
  let fib2 = 0 // (K-2)'th Fibonacci Number
  let fib1 = 1 // (K-1)'th Fibonacci Number.
  let fibK = fib2 + fib1 // Kth Fibonacci

  /* We want to store the smallest fibonacci number smaller such that
    number is greater than or equal to n, we use fibK for this */
  while (fibK < n) {
    fib2 = fib1
    fib1 = fibK
    fibK = fib2 + fib1
  }
  //  This marks the eliminated range from front
  let offset = -1

  /* while there are elements to be checked. We compare arr[fib2] with x.
    When fibM becomes 1, fib2 becomes 0 */

  while (fibK > 1) {
    // Check if fibK is a valid location
    const i = Math.min(offset + fib2, n - 1)

    /*  If x is greater than the value at
      index fib2, Partition the subarray array
      from offset to i */
    if (arr[i] < x) {
      fibK = fib1
      fib1 = fib2
      fib2 = fibK - fib1
      offset = i
      /* If x is greater than the value at
            index fib2, cut the subarray array
            from offset to i */
    } else if (arr[i] > x) {
      fibK = fib2
      fib1 = fib1 - fib2
      fib2 = fibK - fib1
    } else {
    //  return index for found element
      return i
    }
  }

  //    comparing the last element with x */
  if (fib1 && arr[offset + 1] === x) {
    return offset + 1
  }
  //    element not found. return -1
  return -1
}

// Example
// const myArray = [10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100]
// const n = myArray.length
// const x = 90
// const fibFinder = fibonacciSearch(myArray, x, n)
    `,
    java: `
    package com.thealgorithms.searches;

import com.thealgorithms.devutils.searches.SearchAlgorithm;

/*
 *  Fibonacci Search is a popular algorithm which finds the position of a target value in
 *  a sorted array
 *
 *  The time complexity for this search algorithm is O(log3(n))
 *  The space complexity for this search algorithm is O(1)
 *  @author Kanakalatha Vemuru (https://github.com/KanakalathaVemuru)
 */
public class FibonacciSearch implements SearchAlgorithm {

    /**
     * @param array is a sorted array where the element has to be searched
     * @param key is an element whose position has to be found
     * @param <T> is any comparable type
     * @return index of the element
     */
    @Override
    public <T extends Comparable<T>> int find(T[] array, T key) {
        int fibMinus1 = 1;
        int fibMinus2 = 0;
        int fibNumber = fibMinus1 + fibMinus2;
        int n = array.length;

        while (fibNumber < n) {
            fibMinus2 = fibMinus1;
            fibMinus1 = fibNumber;
            fibNumber = fibMinus2 + fibMinus1;
        }

        int offset = -1;

        while (fibNumber > 1) {
            int i = Math.min(offset + fibMinus2, n - 1);

            if (array[i].compareTo(key) < 0) {
                fibNumber = fibMinus1;
                fibMinus1 = fibMinus2;
                fibMinus2 = fibNumber - fibMinus1;
                offset = i;
            } else if (array[i].compareTo(key) > 0) {
                fibNumber = fibMinus2;
                fibMinus1 = fibMinus1 - fibMinus2;
                fibMinus2 = fibNumber - fibMinus1;
            } else {
                return i;
            }
        }

        if (fibMinus1 == 1 && array[offset + 1] == key) {
            return offset + 1;
        }

        return -1;
    }

    // Driver Program
    public static void main(String[] args) {
        Integer[] integers = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

        int size = integers.length;
        Integer shouldBeFound = 128;
        FibonacciSearch fsearch = new FibonacciSearch();
        int atIndex = fsearch.find(integers, shouldBeFound);

        System.out.println("Should be found: " + shouldBeFound + ". Found " + integers[atIndex] + " at index " + atIndex + ". An array length " + size);
    }
}

    `,
  },
  {
    id: "8",
    type: "data structures",
    name: "permutations",
    python: `
    def permute_recursive(nums: list[int]) -> list[list[int]]:
    """
    Return all permutations.

    >>> permute_recursive([1, 2, 3])
    [[3, 2, 1], [2, 3, 1], [1, 3, 2], [3, 1, 2], [2, 1, 3], [1, 2, 3]]
    """
    result: list[list[int]] = []
    if len(nums) == 0:
        return [[]]
    for _ in range(len(nums)):
        n = nums.pop(0)
        permutations = permute_recursive(nums.copy())
        for perm in permutations:
            perm.append(n)
        result.extend(permutations)
        nums.append(n)
    return result


def permute_backtrack(nums: list[int]) -> list[list[int]]:
    """
    Return all permutations of the given list.

    >>> permute_backtrack([1, 2, 3])
    [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]
    """

    def backtrack(start: int) -> None:
        if start == len(nums) - 1:
            output.append(nums[:])
        else:
            for i in range(start, len(nums)):
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]  # backtrack

    output: list[list[int]] = []
    backtrack(0)
    return output


if __name__ == "__main__":
    import doctest

    result = permute_backtrack([1, 2, 3])
    print(result)
    doctest.testmod()
    `,
    js: "js",
    java: "java",
  },
  {
    id: "9",
    type: "data structures",
    name: "prefix sum",
    python: `
    """
Author  : Alexander Pantyukhin
Date    : November 3, 2022

Implement the class of prefix sum with useful functions based on it.

"""


class PrefixSum:
    def __init__(self, array: list[int]) -> None:
        len_array = len(array)
        self.prefix_sum = [0] * len_array

        if len_array > 0:
            self.prefix_sum[0] = array[0]

        for i in range(1, len_array):
            self.prefix_sum[i] = self.prefix_sum[i - 1] + array[i]

    def get_sum(self, start: int, end: int) -> int:
        """
        The function returns the sum of array from the start to the end indexes.
        Runtime : O(1)
        Space: O(1)

        >>> PrefixSum([1,2,3]).get_sum(0, 2)
        6
        >>> PrefixSum([1,2,3]).get_sum(1, 2)
        5
        >>> PrefixSum([1,2,3]).get_sum(2, 2)
        3
        >>> PrefixSum([1,2,3]).get_sum(2, 3)
        Traceback (most recent call last):
        ...
        IndexError: list index out of range
        """
        if start == 0:
            return self.prefix_sum[end]

        return self.prefix_sum[end] - self.prefix_sum[start - 1]

    def contains_sum(self, target_sum: int) -> bool:
        """
        The function returns True if array contains the target_sum,
        False otherwise.

        Runtime : O(n)
        Space: O(n)

        >>> PrefixSum([1,2,3]).contains_sum(6)
        True
        >>> PrefixSum([1,2,3]).contains_sum(5)
        True
        >>> PrefixSum([1,2,3]).contains_sum(3)
        True
        >>> PrefixSum([1,2,3]).contains_sum(4)
        False
        >>> PrefixSum([1,2,3]).contains_sum(7)
        False
        >>> PrefixSum([1,-2,3]).contains_sum(2)
        True
        """

        sums = {0}
        for sum_item in self.prefix_sum:
            if sum_item - target_sum in sums:
                return True

            sums.add(sum_item)

        return False


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    `,
    js: "js",
    java: "java",
  },
  {
    id: "10",
    type: "data structures",
    name: "product sum",
    python: `
    """
Calculate the Product Sum from a Special Array.
reference: https://dev.to/sfrasica/algorithms-product-sum-from-an-array-dc6

Python doctests can be run with the following command:
python -m doctest -v product_sum.py

Calculate the product sum of a "special" array which can contain integers or nested
arrays. The product sum is obtained by adding all elements and multiplying by their
respective depths.

For example, in the array [x, y], the product sum is (x + y). In the array [x, [y, z]],
the product sum is x + 2 * (y + z). In the array [x, [y, [z]]],
the product sum is x + 2 * (y + 3z).

Example Input:
[5, 2, [-7, 1], 3, [6, [-13, 8], 4]]
Output: 12

"""


def product_sum(arr: list[int | list], depth: int) -> int:
    """
    Recursively calculates the product sum of an array.

    The product sum of an array is defined as the sum of its elements multiplied by
    their respective depths.  If an element is a list, its product sum is calculated
    recursively by multiplying the sum of its elements with its depth plus one.

    Args:
        arr: The array of integers and nested lists.
        depth: The current depth level.

    Returns:
        int: The product sum of the array.

    Examples:
        >>> product_sum([1, 2, 3], 1)
        6
        >>> product_sum([-1, 2, [-3, 4]], 2)
        8
        >>> product_sum([1, 2, 3], -1)
        -6
        >>> product_sum([1, 2, 3], 0)
        0
        >>> product_sum([1, 2, 3], 7)
        42
        >>> product_sum((1, 2, 3), 7)
        42
        >>> product_sum({1, 2, 3}, 7)
        42
        >>> product_sum([1, -1], 1)
        0
        >>> product_sum([1, -2], 1)
        -1
        >>> product_sum([-3.5, [1, [0.5]]], 1)
        1.5

    """
    total_sum = 0
    for ele in arr:
        total_sum += product_sum(ele, depth + 1) if isinstance(ele, list) else ele
    return total_sum * depth


def product_sum_array(array: list[int | list]) -> int:
    """
    Calculates the product sum of an array.

    Args:
        array (List[Union[int, List]]): The array of integers and nested lists.

    Returns:
        int: The product sum of the array.

    Examples:
        >>> product_sum_array([1, 2, 3])
        6
        >>> product_sum_array([1, [2, 3]])
        11
        >>> product_sum_array([1, [2, [3, 4]]])
        47
        >>> product_sum_array([0])
        0
        >>> product_sum_array([-3.5, [1, [0.5]]])
        1.5
        >>> product_sum_array([1, -2])
        -1

    """
    return product_sum(array, 1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    `,
    js: "js",
    java: "java",
  },
  {
    id: "11",
    type: "data structures",
    name: "AVL tree",
    python: `
    """
Implementation of an auto-balanced binary tree!
For doctests run following command:
python3 -m doctest -v avl_tree.py
For testing run:
python avl_tree.py
"""
from __future__ import annotations

import math
import random
from typing import Any


class MyQueue:
    def __init__(self) -> None:
        self.data: list[Any] = []
        self.head: int = 0
        self.tail: int = 0

    def is_empty(self) -> bool:
        return self.head == self.tail

    def push(self, data: Any) -> None:
        self.data.append(data)
        self.tail = self.tail + 1

    def pop(self) -> Any:
        ret = self.data[self.head]
        self.head = self.head + 1
        return ret

    def count(self) -> int:
        return self.tail - self.head

    def print_queue(self) -> None:
        print(self.data)
        print("**************")
        print(self.data[self.head : self.tail])


class MyNode:
    def __init__(self, data: Any) -> None:
        self.data = data
        self.left: MyNode | None = None
        self.right: MyNode | None = None
        self.height: int = 1

    def get_data(self) -> Any:
        return self.data

    def get_left(self) -> MyNode | None:
        return self.left

    def get_right(self) -> MyNode | None:
        return self.right

    def get_height(self) -> int:
        return self.height

    def set_data(self, data: Any) -> None:
        self.data = data

    def set_left(self, node: MyNode | None) -> None:
        self.left = node

    def set_right(self, node: MyNode | None) -> None:
        self.right = node

    def set_height(self, height: int) -> None:
        self.height = height


def get_height(node: MyNode | None) -> int:
    if node is None:
        return 0
    return node.get_height()


def my_max(a: int, b: int) -> int:
    if a > b:
        return a
    return b


def right_rotation(node: MyNode) -> MyNode:
    r"""
            A                      B
           / \                    / \
          B   C                  Bl  A
         / \       -->          /   / \
        Bl  Br                 UB Br  C
       /
     UB
    UB = unbalanced node
    """
    print("left rotation node:", node.get_data())
    ret = node.get_left()
    assert ret is not None
    node.set_left(ret.get_right())
    ret.set_right(node)
    h1 = my_max(get_height(node.get_right()), get_height(node.get_left())) + 1
    node.set_height(h1)
    h2 = my_max(get_height(ret.get_right()), get_height(ret.get_left())) + 1
    ret.set_height(h2)
    return ret


def left_rotation(node: MyNode) -> MyNode:
    """
    a mirror symmetry rotation of the left_rotation
    """
    print("right rotation node:", node.get_data())
    ret = node.get_right()
    assert ret is not None
    node.set_right(ret.get_left())
    ret.set_left(node)
    h1 = my_max(get_height(node.get_right()), get_height(node.get_left())) + 1
    node.set_height(h1)
    h2 = my_max(get_height(ret.get_right()), get_height(ret.get_left())) + 1
    ret.set_height(h2)
    return ret


def lr_rotation(node: MyNode) -> MyNode:
    r"""
            A              A                    Br
           / \            / \                  /  \
          B   C    LR    Br  C       RR       B    A
         / \       -->  /  \         -->    /     / \
        Bl  Br         B   UB              Bl    UB  C
             \        /
             UB     Bl
    RR = right_rotation   LR = left_rotation
    """
    left_child = node.get_left()
    assert left_child is not None
    node.set_left(left_rotation(left_child))
    return right_rotation(node)


def rl_rotation(node: MyNode) -> MyNode:
    right_child = node.get_right()
    assert right_child is not None
    node.set_right(right_rotation(right_child))
    return left_rotation(node)


def insert_node(node: MyNode | None, data: Any) -> MyNode | None:
    if node is None:
        return MyNode(data)
    if data < node.get_data():
        node.set_left(insert_node(node.get_left(), data))
        if (
            get_height(node.get_left()) - get_height(node.get_right()) == 2
        ):  # an unbalance detected
            left_child = node.get_left()
            assert left_child is not None
            if (
                data < left_child.get_data()
            ):  # new node is the left child of the left child
                node = right_rotation(node)
            else:
                node = lr_rotation(node)
    else:
        node.set_right(insert_node(node.get_right(), data))
        if get_height(node.get_right()) - get_height(node.get_left()) == 2:
            right_child = node.get_right()
            assert right_child is not None
            if data < right_child.get_data():
                node = rl_rotation(node)
            else:
                node = left_rotation(node)
    h1 = my_max(get_height(node.get_right()), get_height(node.get_left())) + 1
    node.set_height(h1)
    return node


def get_right_most(root: MyNode) -> Any:
    while True:
        right_child = root.get_right()
        if right_child is None:
            break
        root = right_child
    return root.get_data()


def get_left_most(root: MyNode) -> Any:
    while True:
        left_child = root.get_left()
        if left_child is None:
            break
        root = left_child
    return root.get_data()


def del_node(root: MyNode, data: Any) -> MyNode | None:
    left_child = root.get_left()
    right_child = root.get_right()
    if root.get_data() == data:
        if left_child is not None and right_child is not None:
            temp_data = get_left_most(right_child)
            root.set_data(temp_data)
            root.set_right(del_node(right_child, temp_data))
        elif left_child is not None:
            root = left_child
        elif right_child is not None:
            root = right_child
        else:
            return None
    elif root.get_data() > data:
        if left_child is None:
            print("No such data")
            return root
        else:
            root.set_left(del_node(left_child, data))
    else:  # root.get_data() < data
        if right_child is None:
            return root
        else:
            root.set_right(del_node(right_child, data))

    if get_height(right_child) - get_height(left_child) == 2:
        assert right_child is not None
        if get_height(right_child.get_right()) > get_height(right_child.get_left()):
            root = left_rotation(root)
        else:
            root = rl_rotation(root)
    elif get_height(right_child) - get_height(left_child) == -2:
        assert left_child is not None
        if get_height(left_child.get_left()) > get_height(left_child.get_right()):
            root = right_rotation(root)
        else:
            root = lr_rotation(root)
    height = my_max(get_height(root.get_right()), get_height(root.get_left())) + 1
    root.set_height(height)
    return root


class AVLtree:
    """
    An AVL tree doctest
    Examples:
    >>> t = AVLtree()
    >>> t.insert(4)
    insert:4
    >>> print(str(t).replace(" \\n","\\n"))
     4
    *************************************
    >>> t.insert(2)
    insert:2
    >>> print(str(t).replace(" \\n","\\n").replace(" \\n","\\n"))
      4
     2  *
    *************************************
    >>> t.insert(3)
    insert:3
    right rotation node: 2
    left rotation node: 4
    >>> print(str(t).replace(" \\n","\\n").replace(" \\n","\\n"))
      3
     2  4
    *************************************
    >>> t.get_height()
    2
    >>> t.del_node(3)
    delete:3
    >>> print(str(t).replace(" \\n","\\n").replace(" \\n","\\n"))
      4
     2  *
    *************************************
    """

    def __init__(self) -> None:
        self.root: MyNode | None = None

    def get_height(self) -> int:
        return get_height(self.root)

    def insert(self, data: Any) -> None:
        print("insert:" + str(data))
        self.root = insert_node(self.root, data)

    def del_node(self, data: Any) -> None:
        print("delete:" + str(data))
        if self.root is None:
            print("Tree is empty!")
            return
        self.root = del_node(self.root, data)

    def __str__(
        self,
    ) -> str:  # a level traversale, gives a more intuitive look on the tree
        output = ""
        q = MyQueue()
        q.push(self.root)
        layer = self.get_height()
        if layer == 0:
            return output
        cnt = 0
        while not q.is_empty():
            node = q.pop()
            space = " " * int(math.pow(2, layer - 1))
            output += space
            if node is None:
                output += "*"
                q.push(None)
                q.push(None)
            else:
                output += str(node.get_data())
                q.push(node.get_left())
                q.push(node.get_right())
            output += space
            cnt = cnt + 1
            for i in range(100):
                if cnt == math.pow(2, i) - 1:
                    layer = layer - 1
                    if layer == 0:
                        output += "\n*************************************"
                        return output
                    output += "\n"
                    break
        output += "\n*************************************"
        return output


def _test() -> None:
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
    t = AVLtree()
    lst = list(range(10))
    random.shuffle(lst)
    for i in lst:
        t.insert(i)
        print(str(t))
    random.shuffle(lst)
    for i in lst:
        t.del_node(i)
        print(str(t))
    `,
    js: `
    /**
 * Adelson-Velsky and Landis Tree
 * [Wikipedia](https://en.wikipedia.org/wiki/AVL_tree)
 * [A video lecture](http://www.youtube.com/watch?v=TbvhGcf6UJU)
 */
'use strict'

/**
 * A utility class for comparator
 * A comparator is expected to have following structure
 *
 * comp(a, b) RETURN < 0 if a < b
 * RETURN > 0 if a > b
 * MUST RETURN 0 if a == b
 */
let utils;
(function (_utils) {
  function comparator () {
    return function (v1, v2) {
      if (v1 < v2) return -1
      if (v2 < v1) return 1
      return 0
    }
  }
  _utils.comparator = comparator
})(utils || (utils = {}))

/**
 * @constructor
 * A class for AVL Tree
 * @argument comp - A function used by AVL Tree For Comparison
 * If no argument is sent it uses utils.comparator
 */
const AVLTree = (function () {
  function _avl (comp) {
    /** @public comparator function */
    this._comp = undefined
    this._comp = comp !== undefined ? comp : utils.comparator()

    /** @public root of the AVL Tree */
    this.root = null
    /** @public number of elements in AVL Tree */
    this.size = 0
  }

  // creates new Node Object
  const Node = function (val) {
    this._val = val
    this._left = null
    this._right = null
    this._height = 1
  }

  // get height of a node
  const getHeight = function (node) {
    if (node == null) { return 0 }
    return node._height
  }

  // height difference or balance factor of a node
  const getHeightDifference = function (node) {
    return node == null ? 0 : getHeight(node._left) - getHeight(node._right)
  }

  // update height of a node based on children's heights
  const updateHeight = function (node) {
    if (node == null) { return }
    node._height = Math.max(getHeight(node._left), getHeight(node._right)) + 1
  }

  // Helper: To check if the balanceFactor is valid
  const isValidBalanceFactor = (balanceFactor) => [0, 1, -1].includes(balanceFactor)

  // rotations of AVL Tree
  const leftRotate = function (node) {
    const temp = node._right
    node._right = temp._left
    temp._left = node
    updateHeight(node)
    updateHeight(temp)
    return temp
  }
  const rightRotate = function (node) {
    const temp = node._left
    node._left = temp._right
    temp._right = node
    updateHeight(node)
    updateHeight(temp)
    return temp
  }

  // check if tree is balanced else balance it for insertion
  const insertBalance = function (node, _val, balanceFactor, tree) {
    if (balanceFactor > 1 && tree._comp(_val, node._left._val) < 0) {
      return rightRotate(node) // Left Left Case
    }
    if (balanceFactor < 1 && tree._comp(_val, node._right._val) > 0) {
      return leftRotate(node) // Right Right Case
    }
    if (balanceFactor > 1 && tree._comp(_val, node._left._val) > 0) {
      node._left = leftRotate(node._left) // Left Right Case
      return rightRotate(node)
    }
    node._right = rightRotate(node._right)
    return leftRotate(node)
  }

  // check if tree is balanced after deletion
  const delBalance = function (node) {
    const balanceFactor1 = getHeightDifference(node)
    if (isValidBalanceFactor(balanceFactor1)) {
      return node
    }
    if (balanceFactor1 > 1) {
      if (getHeightDifference(node._left) >= 0) {
        return rightRotate(node) // Left Left
      }
      node._left = leftRotate(node._left)
      return rightRotate(node) // Left Right
    }
    if (getHeightDifference(node._right) > 0) {
      node._right = rightRotate(node._right)
      return leftRotate(node) // Right Left
    }
    return leftRotate(node) // Right Right
  }

  // implement avl tree insertion
  const insert = function (root, val, tree) {
    if (root == null) {
      tree.size++
      return new Node(val)
    }
    if (tree._comp(root._val, val) < 0) {
      root._right = insert(root._right, val, tree)
    } else if (tree._comp(root._val, val) > 0) {
      root._left = insert(root._left, val, tree)
    } else {
      return root
    }
    updateHeight(root)
    const balanceFactor = getHeightDifference(root)
    return isValidBalanceFactor(balanceFactor) ? root : insertBalance(root, val, balanceFactor, tree)
  }

  // delete am element
  const deleteElement = function (root, _val, tree) {
    if (root == null) { return root }
    if (tree._comp(root._val, _val) === 0) { // key found case
      if (root._left === null && root._right === null) {
        root = null
        tree.size--
      } else if (root._left === null) {
        root = root._right
        tree.size--
      } else if (root._right === null) {
        root = root._left
        tree.size--
      } else {
        let temp = root._right
        while (temp._left != null) {
          temp = temp._left
        }
        root._val = temp._val
        root._right = deleteElement(root._right, temp._val, tree)
      }
    } else {
      if (tree._comp(root._val, _val) < 0) {
        root._right = deleteElement(root._right, _val, tree)
      } else {
        root._left = deleteElement(root._left, _val, tree)
      }
    }
    updateHeight(root)
    root = delBalance(root)
    return root
  }
  // search tree for a element
  const searchAVLTree = function (root, val, tree) {
    if (root == null) { return null }
    if (tree._comp(root._val, val) === 0) {
      return root
    }
    if (tree._comp(root._val, val) < 0) {
      return searchAVLTree(root._right, val, tree)
    }
    return searchAVLTree(root._left, val, tree)
  }

  /* Public Functions */
  /**
   * For Adding Elements to AVL Tree
   * @param {any} _val
   * Since in AVL Tree an element can only occur once so
   * if a element exists it return false
   * @returns {Boolean} element added or not
   */
  _avl.prototype.add = function (_val) {
    const prevSize = this.size
    this.root = insert(this.root, _val, this)
    return this.size !== prevSize
  }
  /**
   * TO check is a particular element exists or not
   * @param {any} _val
   * @returns {Boolean} exists or not
   */
  _avl.prototype.find = function (_val) {
    const temp = searchAVLTree(this.root, _val, this)
    return temp != null
  }
  /**
   *
   * @param {any} _val
   * It is possible that element doesn't exists in tree
   * in that case it return false
   * @returns {Boolean} if element was found and deleted
   */
  _avl.prototype.remove = function (_val) {
    const prevSize = this.size
    this.root = deleteElement(this.root, _val, this)
    return prevSize !== this.size
  }
  return _avl
}())

/**
 * A Code for Testing the AVLTree
 */
// (function test () {
//   const newAVL = new AVLTree()
//   const size = Math.floor(Math.random() * 1000000)
//   let uniques = 0
//   let i, temp, j
//   const array = []
//   for (i = 0; i < size; i++) {
//     temp = Math.floor(Math.random() * Number.MAX_VALUE)
//     if (newAVL.add(temp)) {
//       uniques++
//       array.push(temp)
//     }
//   }
//   if (newAVL.size !== uniques) {
//     throw new Error('elements not inserted properly')
//   }
//   const findTestSize = Math.floor(Math.random() * uniques)
//   for (i = 0; i < findTestSize; i++) {
//     j = Math.floor(Math.random() * uniques)
//     if (!newAVL.find(array[j])) {
//       throw new Error('inserted elements not found')
//     }
//   }
//   const deleteTestSize = Math.floor(uniques * Math.random())
//   for (i = 0; i < deleteTestSize; i++) {
//     j = Math.floor(Math.random() * uniques)
//     temp = array[j]
//     if (newAVL.find(temp)) {
//       if (!newAVL.remove(temp)) {
//         throw new Error('delete not working properly')
//       }
//     }
//   }
// })()

export { AVLTree }
    `,
    java: `
    package com.thealgorithms.datastructures.trees;

public class AVLTree {

    private Node root;

    private class Node {

        private int key;
        private int balance;
        private int height;
        private Node left, right, parent;

        Node(int k, Node p) {
            key = k;
            parent = p;
        }
    }

    public boolean insert(int key) {
        if (root == null) {
            root = new Node(key, null);
        } else {
            Node n = root;
            Node parent;
            while (true) {
                if (n.key == key) {
                    return false;
                }

                parent = n;

                boolean goLeft = n.key > key;
                n = goLeft ? n.left : n.right;

                if (n == null) {
                    if (goLeft) {
                        parent.left = new Node(key, parent);
                    } else {
                        parent.right = new Node(key, parent);
                    }
                    rebalance(parent);
                    break;
                }
            }
        }
        return true;
    }

    private void delete(Node node) {
        if (node.left == null && node.right == null) {
            if (node.parent == null) {
                root = null;
            } else {
                Node parent = node.parent;
                if (parent.left == node) {
                    parent.left = null;
                } else {
                    parent.right = null;
                }
                rebalance(parent);
            }
            return;
        }
        Node child;
        if (node.left != null) {
            child = node.left;
            while (child.right != null) {
                child = child.right;
            }
        } else {
            child = node.right;
            while (child.left != null) {
                child = child.left;
            }
        }
        node.key = child.key;
        delete(child);
    }

    public void delete(int delKey) {
        if (root == null) {
            return;
        }
        Node node = root;
        Node child = root;

        while (child != null) {
            node = child;
            child = delKey >= node.key ? node.right : node.left;
            if (delKey == node.key) {
                delete(node);
                return;
            }
        }
    }

    private void rebalance(Node n) {
        setBalance(n);

        if (n.balance == -2) {
            if (height(n.left.left) >= height(n.left.right)) {
                n = rotateRight(n);
            } else {
                n = rotateLeftThenRight(n);
            }
        } else if (n.balance == 2) {
            if (height(n.right.right) >= height(n.right.left)) {
                n = rotateLeft(n);
            } else {
                n = rotateRightThenLeft(n);
            }
        }

        if (n.parent != null) {
            rebalance(n.parent);
        } else {
            root = n;
        }
    }

    private Node rotateLeft(Node a) {
        Node b = a.right;
        b.parent = a.parent;

        a.right = b.left;

        if (a.right != null) {
            a.right.parent = a;
        }

        b.left = a;
        a.parent = b;

        if (b.parent != null) {
            if (b.parent.right == a) {
                b.parent.right = b;
            } else {
                b.parent.left = b;
            }
        }

        setBalance(a, b);

        return b;
    }

    private Node rotateRight(Node a) {
        Node b = a.left;
        b.parent = a.parent;

        a.left = b.right;

        if (a.left != null) {
            a.left.parent = a;
        }

        b.right = a;
        a.parent = b;

        if (b.parent != null) {
            if (b.parent.right == a) {
                b.parent.right = b;
            } else {
                b.parent.left = b;
            }
        }

        setBalance(a, b);

        return b;
    }

    private Node rotateLeftThenRight(Node n) {
        n.left = rotateLeft(n.left);
        return rotateRight(n);
    }

    private Node rotateRightThenLeft(Node n) {
        n.right = rotateRight(n.right);
        return rotateLeft(n);
    }

    private int height(Node n) {
        if (n == null) {
            return -1;
        }
        return n.height;
    }

    private void setBalance(Node... nodes) {
        for (Node n : nodes) {
            reheight(n);
            n.balance = height(n.right) - height(n.left);
        }
    }

    public void printBalance() {
        printBalance(root);
    }

    private void printBalance(Node n) {
        if (n != null) {
            printBalance(n.left);
            System.out.printf("%s ", n.balance);
            printBalance(n.right);
        }
    }

    private void reheight(Node node) {
        if (node != null) {
            node.height = 1 + Math.max(height(node.left), height(node.right));
        }
    }

    public boolean search(int key) {
        Node result = searchHelper(this.root, key);
        return result != null;
    }

    private Node searchHelper(Node root, int key) {
        // root is null or key is present at root
        if (root == null || root.key == key) {
            return root;
        }

        // key is greater than root's key
        if (root.key > key) {
            return searchHelper(root.left, key); // call the function on the node's left child
        }
        // key is less than root's key then
        // call the function on the node's right child as it is greater
        return searchHelper(root.right, key);
    }

    public static void main(String[] args) {
        AVLTree tree = new AVLTree();

        System.out.println("Inserting values 1 to 10");
        for (int i = 1; i < 10; i++) {
            tree.insert(i);
        }

        System.out.print("Printing balance: ");
        tree.printBalance();
    }
}
    `,
  },
  {
    id: "12",
    type: "mathematics",
    name: "abs",
    python: `
    """Absolute Value."""


def abs_val(num: float) -> float:
    """
    Find the absolute value of a number.

    >>> abs_val(-5.1)
    5.1
    >>> abs_val(-5) == abs_val(5)
    True
    >>> abs_val(0)
    0
    """
    return -num if num < 0 else num


def abs_min(x: list[int]) -> int:
    """
    >>> abs_min([0,5,1,11])
    0
    >>> abs_min([3,-10,-2])
    -2
    >>> abs_min([])
    Traceback (most recent call last):
        ...
    ValueError: abs_min() arg is an empty sequence
    """
    if len(x) == 0:
        raise ValueError("abs_min() arg is an empty sequence")
    j = x[0]
    for i in x:
        if abs_val(i) < abs_val(j):
            j = i
    return j


def abs_max(x: list[int]) -> int:
    """
    >>> abs_max([0,5,1,11])
    11
    >>> abs_max([3,-10,-2])
    -10
    >>> abs_max([])
    Traceback (most recent call last):
        ...
    ValueError: abs_max() arg is an empty sequence
    """
    if len(x) == 0:
        raise ValueError("abs_max() arg is an empty sequence")
    j = x[0]
    for i in x:
        if abs(i) > abs(j):
            j = i
    return j


def abs_max_sort(x: list[int]) -> int:
    """
    >>> abs_max_sort([0,5,1,11])
    11
    >>> abs_max_sort([3,-10,-2])
    -10
    >>> abs_max_sort([])
    Traceback (most recent call last):
        ...
    ValueError: abs_max_sort() arg is an empty sequence
    """
    if len(x) == 0:
        raise ValueError("abs_max_sort() arg is an empty sequence")
    return sorted(x, key=abs)[-1]


def test_abs_val():
    """
    >>> test_abs_val()
    """
    assert abs_val(0) == 0
    assert abs_val(34) == 34
    assert abs_val(-100000000000) == 100000000000

    a = [-3, -1, 2, -11]
    assert abs_max(a) == -11
    assert abs_max_sort(a) == -11
    assert abs_min(a) == -1


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    test_abs_val()
    print(abs_val(-34))  # --> 34
    `,
    js: `
    /**
 * @function abs
 * @description This script will find the absolute value of a number.
 * @param {number} num - The input integer
 * @return {number} - Absolute number of num.
 * @see https://en.wikipedia.org/wiki/Absolute_value
 * @example abs(-10) = 10
 * @example abs(50) = 50
 * @example abs(0) = 0
 */

const abs = (num) => {
  const validNumber = +num // converted to number, also can use - Number(num)

  if (Number.isNaN(validNumber)) {
    throw new TypeError('Argument is NaN - Not a Number')
  }

  return validNumber < 0 ? -validNumber : validNumber // if number is less then zero mean negative then it converted to positive. i.e -> n = -2 = -(-2) = 2
}

export { abs }
    `,
    java: "java",
  },
  {
    id: "13",
    type: "mathematics",
    name: "area",
    python: `
    """
Find the area of various geometric shapes
Wikipedia reference: https://en.wikipedia.org/wiki/Area
"""
from math import pi, sqrt, tan


def surface_area_cube(side_length: float) -> float:
    """
    Calculate the Surface Area of a Cube.

    >>> surface_area_cube(1)
    6
    >>> surface_area_cube(1.6)
    15.360000000000003
    >>> surface_area_cube(0)
    0
    >>> surface_area_cube(3)
    54
    >>> surface_area_cube(-1)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cube() only accepts non-negative values
    """
    if side_length < 0:
        raise ValueError("surface_area_cube() only accepts non-negative values")
    return 6 * side_length**2


def surface_area_cuboid(length: float, breadth: float, height: float) -> float:
    """
    Calculate the Surface Area of a Cuboid.

    >>> surface_area_cuboid(1, 2, 3)
    22
    >>> surface_area_cuboid(0, 0, 0)
    0
    >>> surface_area_cuboid(1.6, 2.6, 3.6)
    38.56
    >>> surface_area_cuboid(-1, 2, 3)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cuboid() only accepts non-negative values
    >>> surface_area_cuboid(1, -2, 3)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cuboid() only accepts non-negative values
    >>> surface_area_cuboid(1, 2, -3)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cuboid() only accepts non-negative values
    """
    if length < 0 or breadth < 0 or height < 0:
        raise ValueError("surface_area_cuboid() only accepts non-negative values")
    return 2 * ((length * breadth) + (breadth * height) + (length * height))


def surface_area_sphere(radius: float) -> float:
    """
    Calculate the Surface Area of a Sphere.
    Wikipedia reference: https://en.wikipedia.org/wiki/Sphere
    Formula: 4 * pi * r^2

    >>> surface_area_sphere(5)
    314.1592653589793
    >>> surface_area_sphere(1)
    12.566370614359172
    >>> surface_area_sphere(1.6)
    32.169908772759484
    >>> surface_area_sphere(0)
    0.0
    >>> surface_area_sphere(-1)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_sphere() only accepts non-negative values
    """
    if radius < 0:
        raise ValueError("surface_area_sphere() only accepts non-negative values")
    return 4 * pi * radius**2


def surface_area_hemisphere(radius: float) -> float:
    """
    Calculate the Surface Area of a Hemisphere.
    Formula: 3 * pi * r^2

    >>> surface_area_hemisphere(5)
    235.61944901923448
    >>> surface_area_hemisphere(1)
    9.42477796076938
    >>> surface_area_hemisphere(0)
    0.0
    >>> surface_area_hemisphere(1.1)
    11.40398133253095
    >>> surface_area_hemisphere(-1)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_hemisphere() only accepts non-negative values
    """
    if radius < 0:
        raise ValueError("surface_area_hemisphere() only accepts non-negative values")
    return 3 * pi * radius**2


def surface_area_cone(radius: float, height: float) -> float:
    """
    Calculate the Surface Area of a Cone.
    Wikipedia reference: https://en.wikipedia.org/wiki/Cone
    Formula: pi * r * (r + (h ** 2 + r ** 2) ** 0.5)

    >>> surface_area_cone(10, 24)
    1130.9733552923256
    >>> surface_area_cone(6, 8)
    301.59289474462014
    >>> surface_area_cone(1.6, 2.6)
    23.387862992395807
    >>> surface_area_cone(0, 0)
    0.0
    >>> surface_area_cone(-1, -2)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cone() only accepts non-negative values
    >>> surface_area_cone(1, -2)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cone() only accepts non-negative values
    >>> surface_area_cone(-1, 2)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cone() only accepts non-negative values
    """
    if radius < 0 or height < 0:
        raise ValueError("surface_area_cone() only accepts non-negative values")
    return pi * radius * (radius + (height**2 + radius**2) ** 0.5)


def surface_area_conical_frustum(
    radius_1: float, radius_2: float, height: float
) -> float:
    """
    Calculate the Surface Area of a Conical Frustum.

    >>> surface_area_conical_frustum(1, 2, 3)
    45.511728065337266
    >>> surface_area_conical_frustum(4, 5, 6)
    300.7913575056268
    >>> surface_area_conical_frustum(0, 0, 0)
    0.0
    >>> surface_area_conical_frustum(1.6, 2.6, 3.6)
    78.57907060751548
    >>> surface_area_conical_frustum(-1, 2, 3)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_conical_frustum() only accepts non-negative values
    >>> surface_area_conical_frustum(1, -2, 3)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_conical_frustum() only accepts non-negative values
    >>> surface_area_conical_frustum(1, 2, -3)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_conical_frustum() only accepts non-negative values
    """
    if radius_1 < 0 or radius_2 < 0 or height < 0:
        raise ValueError(
            "surface_area_conical_frustum() only accepts non-negative values"
        )
    slant_height = (height**2 + (radius_1 - radius_2) ** 2) ** 0.5
    return pi * ((slant_height * (radius_1 + radius_2)) + radius_1**2 + radius_2**2)


def surface_area_cylinder(radius: float, height: float) -> float:
    """
    Calculate the Surface Area of a Cylinder.
    Wikipedia reference: https://en.wikipedia.org/wiki/Cylinder
    Formula: 2 * pi * r * (h + r)

    >>> surface_area_cylinder(7, 10)
    747.6990515543707
    >>> surface_area_cylinder(1.6, 2.6)
    42.22300526424682
    >>> surface_area_cylinder(0, 0)
    0.0
    >>> surface_area_cylinder(6, 8)
    527.7875658030853
    >>> surface_area_cylinder(-1, -2)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cylinder() only accepts non-negative values
    >>> surface_area_cylinder(1, -2)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cylinder() only accepts non-negative values
    >>> surface_area_cylinder(-1, 2)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_cylinder() only accepts non-negative values
    """
    if radius < 0 or height < 0:
        raise ValueError("surface_area_cylinder() only accepts non-negative values")
    return 2 * pi * radius * (height + radius)


def surface_area_torus(torus_radius: float, tube_radius: float) -> float:
    """Calculate the Area of a Torus.
    Wikipedia reference: https://en.wikipedia.org/wiki/Torus
    :return 4pi^2 * torus_radius * tube_radius
    >>> surface_area_torus(1, 1)
    39.47841760435743
    >>> surface_area_torus(4, 3)
    473.7410112522892
    >>> surface_area_torus(3, 4)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_torus() does not support spindle or self intersecting tori
    >>> surface_area_torus(1.6, 1.6)
    101.06474906715503
    >>> surface_area_torus(0, 0)
    0.0
    >>> surface_area_torus(-1, 1)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_torus() only accepts non-negative values
    >>> surface_area_torus(1, -1)
    Traceback (most recent call last):
        ...
    ValueError: surface_area_torus() only accepts non-negative values
    """
    if torus_radius < 0 or tube_radius < 0:
        raise ValueError("surface_area_torus() only accepts non-negative values")
    if torus_radius < tube_radius:
        raise ValueError(
            "surface_area_torus() does not support spindle or self intersecting tori"
        )
    return 4 * pow(pi, 2) * torus_radius * tube_radius


def area_rectangle(length: float, width: float) -> float:
    """
    Calculate the area of a rectangle.

    >>> area_rectangle(10, 20)
    200
    >>> area_rectangle(1.6, 2.6)
    4.16
    >>> area_rectangle(0, 0)
    0
    >>> area_rectangle(-1, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_rectangle() only accepts non-negative values
    >>> area_rectangle(1, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_rectangle() only accepts non-negative values
    >>> area_rectangle(-1, 2)
    Traceback (most recent call last):
        ...
    ValueError: area_rectangle() only accepts non-negative values
    """
    if length < 0 or width < 0:
        raise ValueError("area_rectangle() only accepts non-negative values")
    return length * width


def area_square(side_length: float) -> float:
    """
    Calculate the area of a square.

    >>> area_square(10)
    100
    >>> area_square(0)
    0
    >>> area_square(1.6)
    2.5600000000000005
    >>> area_square(-1)
    Traceback (most recent call last):
        ...
    ValueError: area_square() only accepts non-negative values
    """
    if side_length < 0:
        raise ValueError("area_square() only accepts non-negative values")
    return side_length**2


def area_triangle(base: float, height: float) -> float:
    """
    Calculate the area of a triangle given the base and height.

    >>> area_triangle(10, 10)
    50.0
    >>> area_triangle(1.6, 2.6)
    2.08
    >>> area_triangle(0, 0)
    0.0
    >>> area_triangle(-1, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_triangle() only accepts non-negative values
    >>> area_triangle(1, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_triangle() only accepts non-negative values
    >>> area_triangle(-1, 2)
    Traceback (most recent call last):
        ...
    ValueError: area_triangle() only accepts non-negative values
    """
    if base < 0 or height < 0:
        raise ValueError("area_triangle() only accepts non-negative values")
    return (base * height) / 2


def area_triangle_three_sides(side1: float, side2: float, side3: float) -> float:
    """
    Calculate area of triangle when the length of 3 sides are known.
    This function uses Heron's formula: https://en.wikipedia.org/wiki/Heron%27s_formula

    >>> area_triangle_three_sides(5, 12, 13)
    30.0
    >>> area_triangle_three_sides(10, 11, 12)
    51.521233486786784
    >>> area_triangle_three_sides(0, 0, 0)
    0.0
    >>> area_triangle_three_sides(1.6, 2.6, 3.6)
    1.8703742940919619
    >>> area_triangle_three_sides(-1, -2, -1)
    Traceback (most recent call last):
        ...
    ValueError: area_triangle_three_sides() only accepts non-negative values
    >>> area_triangle_three_sides(1, -2, 1)
    Traceback (most recent call last):
        ...
    ValueError: area_triangle_three_sides() only accepts non-negative values
    >>> area_triangle_three_sides(2, 4, 7)
    Traceback (most recent call last):
        ...
    ValueError: Given three sides do not form a triangle
    >>> area_triangle_three_sides(2, 7, 4)
    Traceback (most recent call last):
        ...
    ValueError: Given three sides do not form a triangle
    >>> area_triangle_three_sides(7, 2, 4)
    Traceback (most recent call last):
        ...
    ValueError: Given three sides do not form a triangle
    """
    if side1 < 0 or side2 < 0 or side3 < 0:
        raise ValueError("area_triangle_three_sides() only accepts non-negative values")
    elif side1 + side2 < side3 or side1 + side3 < side2 or side2 + side3 < side1:
        raise ValueError("Given three sides do not form a triangle")
    semi_perimeter = (side1 + side2 + side3) / 2
    area = sqrt(
        semi_perimeter
        * (semi_perimeter - side1)
        * (semi_perimeter - side2)
        * (semi_perimeter - side3)
    )
    return area


def area_parallelogram(base: float, height: float) -> float:
    """
    Calculate the area of a parallelogram.

    >>> area_parallelogram(10, 20)
    200
    >>> area_parallelogram(1.6, 2.6)
    4.16
    >>> area_parallelogram(0, 0)
    0
    >>> area_parallelogram(-1, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_parallelogram() only accepts non-negative values
    >>> area_parallelogram(1, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_parallelogram() only accepts non-negative values
    >>> area_parallelogram(-1, 2)
    Traceback (most recent call last):
        ...
    ValueError: area_parallelogram() only accepts non-negative values
    """
    if base < 0 or height < 0:
        raise ValueError("area_parallelogram() only accepts non-negative values")
    return base * height


def area_trapezium(base1: float, base2: float, height: float) -> float:
    """
    Calculate the area of a trapezium.

    >>> area_trapezium(10, 20, 30)
    450.0
    >>> area_trapezium(1.6, 2.6, 3.6)
    7.5600000000000005
    >>> area_trapezium(0, 0, 0)
    0.0
    >>> area_trapezium(-1, -2, -3)
    Traceback (most recent call last):
        ...
    ValueError: area_trapezium() only accepts non-negative values
    >>> area_trapezium(-1, 2, 3)
    Traceback (most recent call last):
        ...
    ValueError: area_trapezium() only accepts non-negative values
    >>> area_trapezium(1, -2, 3)
    Traceback (most recent call last):
        ...
    ValueError: area_trapezium() only accepts non-negative values
    >>> area_trapezium(1, 2, -3)
    Traceback (most recent call last):
        ...
    ValueError: area_trapezium() only accepts non-negative values
    >>> area_trapezium(-1, -2, 3)
    Traceback (most recent call last):
        ...
    ValueError: area_trapezium() only accepts non-negative values
    >>> area_trapezium(1, -2, -3)
    Traceback (most recent call last):
        ...
    ValueError: area_trapezium() only accepts non-negative values
    >>> area_trapezium(-1, 2, -3)
    Traceback (most recent call last):
        ...
    ValueError: area_trapezium() only accepts non-negative values
    """
    if base1 < 0 or base2 < 0 or height < 0:
        raise ValueError("area_trapezium() only accepts non-negative values")
    return 1 / 2 * (base1 + base2) * height


def area_circle(radius: float) -> float:
    """
    Calculate the area of a circle.

    >>> area_circle(20)
    1256.6370614359173
    >>> area_circle(1.6)
    8.042477193189871
    >>> area_circle(0)
    0.0
    >>> area_circle(-1)
    Traceback (most recent call last):
        ...
    ValueError: area_circle() only accepts non-negative values
    """
    if radius < 0:
        raise ValueError("area_circle() only accepts non-negative values")
    return pi * radius**2


def area_ellipse(radius_x: float, radius_y: float) -> float:
    """
    Calculate the area of a ellipse.

    >>> area_ellipse(10, 10)
    314.1592653589793
    >>> area_ellipse(10, 20)
    628.3185307179587
    >>> area_ellipse(0, 0)
    0.0
    >>> area_ellipse(1.6, 2.6)
    13.06902543893354
    >>> area_ellipse(-10, 20)
    Traceback (most recent call last):
        ...
    ValueError: area_ellipse() only accepts non-negative values
    >>> area_ellipse(10, -20)
    Traceback (most recent call last):
        ...
    ValueError: area_ellipse() only accepts non-negative values
    >>> area_ellipse(-10, -20)
    Traceback (most recent call last):
        ...
    ValueError: area_ellipse() only accepts non-negative values
    """
    if radius_x < 0 or radius_y < 0:
        raise ValueError("area_ellipse() only accepts non-negative values")
    return pi * radius_x * radius_y


def area_rhombus(diagonal_1: float, diagonal_2: float) -> float:
    """
    Calculate the area of a rhombus.

    >>> area_rhombus(10, 20)
    100.0
    >>> area_rhombus(1.6, 2.6)
    2.08
    >>> area_rhombus(0, 0)
    0.0
    >>> area_rhombus(-1, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_rhombus() only accepts non-negative values
    >>> area_rhombus(1, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_rhombus() only accepts non-negative values
    >>> area_rhombus(-1, 2)
    Traceback (most recent call last):
        ...
    ValueError: area_rhombus() only accepts non-negative values
    """
    if diagonal_1 < 0 or diagonal_2 < 0:
        raise ValueError("area_rhombus() only accepts non-negative values")
    return 1 / 2 * diagonal_1 * diagonal_2


def area_reg_polygon(sides: int, length: float) -> float:
    """
    Calculate the area of a regular polygon.
    Wikipedia reference: https://en.wikipedia.org/wiki/Polygon#Regular_polygons
    Formula: (n*s^2*cot(pi/n))/4

    >>> area_reg_polygon(3, 10)
    43.301270189221945
    >>> area_reg_polygon(4, 10)
    100.00000000000001
    >>> area_reg_polygon(0, 0)
    Traceback (most recent call last):
        ...
    ValueError: area_reg_polygon() only accepts integers greater than or equal to \
three as number of sides
    >>> area_reg_polygon(-1, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_reg_polygon() only accepts integers greater than or equal to \
three as number of sides
    >>> area_reg_polygon(5, -2)
    Traceback (most recent call last):
        ...
    ValueError: area_reg_polygon() only accepts non-negative values as \
length of a side
    >>> area_reg_polygon(-1, 2)
    Traceback (most recent call last):
        ...
    ValueError: area_reg_polygon() only accepts integers greater than or equal to \
three as number of sides
    """
    if not isinstance(sides, int) or sides < 3:
        raise ValueError(
            "area_reg_polygon() only accepts integers greater than or \
equal to three as number of sides"
        )
    elif length < 0:
        raise ValueError(
            "area_reg_polygon() only accepts non-negative values as \
length of a side"
        )
    return (sides * length**2) / (4 * tan(pi / sides))
    return (sides * length**2) / (4 * tan(pi / sides))


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)  # verbose so we can see methods missing tests

    print("[DEMO] Areas of various geometric shapes: \n")
    print(f"Rectangle: {area_rectangle(10, 20) = }")
    print(f"Square: {area_square(10) = }")
    print(f"Triangle: {area_triangle(10, 10) = }")
    print(f"Triangle: {area_triangle_three_sides(5, 12, 13) = }")
    print(f"Parallelogram: {area_parallelogram(10, 20) = }")
    print(f"Rhombus: {area_rhombus(10, 20) = }")
    print(f"Trapezium: {area_trapezium(10, 20, 30) = }")
    print(f"Circle: {area_circle(20) = }")
    print(f"Ellipse: {area_ellipse(10, 20) = }")
    print("\nSurface Areas of various geometric shapes: \n")
    print(f"Cube: {surface_area_cube(20) = }")
    print(f"Cuboid: {surface_area_cuboid(10, 20, 30) = }")
    print(f"Sphere: {surface_area_sphere(20) = }")
    print(f"Hemisphere: {surface_area_hemisphere(20) = }")
    print(f"Cone: {surface_area_cone(10, 20) = }")
    print(f"Conical Frustum: {surface_area_conical_frustum(10, 20, 30) = }")
    print(f"Cylinder: {surface_area_cylinder(10, 20) = }")
    print(f"Torus: {surface_area_torus(20, 10) = }")
    print(f"Equilateral Triangle: {area_reg_polygon(3, 10) = }")
    print(f"Square: {area_reg_polygon(4, 10) = }")
    print(f"Reqular Pentagon: {area_reg_polygon(5, 10) = }")
    `,
    js: `
    /*
  Calculate the area of various shapes
*/

/**
 * @function surfaceAreaCube
 * @description Calculate the Surface Area of a Cube.
 * @param {Integer} side - Integer
 * @return {Integer} - 6 * side ** 2
 * @see [surfaceAreaCube](https://en.wikipedia.org/wiki/Area#Surface_area)
 * @example surfaceAreaCube(1) = 6
 */
const surfaceAreaCube = (side) => {
  validateNumericParam(side, 'side')
  return 6 * side ** 2
}

/**
 * @function surfaceAreaSphere
 * @description Calculate the Surface Area of a Sphere.
 * @param {Integer} radius - Integer
 * @return {Integer} - 4 * pi * r^2
 * @see [surfaceAreaSphere](https://en.wikipedia.org/wiki/Sphere)
 * @example surfaceAreaSphere(5) = 314.1592653589793
 */
const surfaceAreaSphere = (radius) => {
  validateNumericParam(radius, 'radius')
  return 4.0 * Math.PI * radius ** 2.0
}

/**
 * @function areaRectangle
 * @description Calculate the area of a rectangle.
 * @param {Integer} length - Integer
 * @param {Integer} width - Integer
 * @return {Integer} - width * length
 * @see [areaRectangle](https://en.wikipedia.org/wiki/Area#Quadrilateral_area)
 * @example areaRectangle(4) = 16
 */
const areaRectangle = (length, width) => {
  validateNumericParam(length, 'Length')
  validateNumericParam(width, 'Width')
  return width * length
}

/**
 * @function areaSquare
 * @description Calculate the area of a square.
 * @param {Integer} side - Integer
 * @return {Integer} - side ** 2.
 * @see [areaSquare](https://en.wikipedia.org/wiki/Square)
 * @example areaSquare(4) = 16
 */
const areaSquare = (side) => {
  validateNumericParam(side, 'square side')
  return side ** 2
}

/**
 * @function areaTriangle
 * @description Calculate the area of a triangle.
 * @param {Integer} base - Integer
 * @param {Integer} height - Integer
 * @return {Integer} - base * height / 2.
 * @see [areaTriangle](https://en.wikipedia.org/wiki/Area#Triangle_area)
 * @example areaTriangle(1.66, 3.44) = 2.8552
 */
const areaTriangle = (base, height) => {
  validateNumericParam(base, 'Base')
  validateNumericParam(height, 'Height')
  return (base * height) / 2.0
}

/**
 * @function areaTriangleWithAllThreeSides
 * @description Calculate the area of a triangle with the all three sides given.
 * @param {Integer} side1 - Integer
 * @param {Integer} side2 - Integer
 * @param {Integer} side3 - Integer
 * @return {Integer} - area of triangle.
 * @see [areaTriangleWithAllThreeSides](https://en.wikipedia.org/wiki/Heron%27s_formula)
 * @example areaTriangleWithAllThreeSides(5, 6, 7) = 14.7
 */
const areaTriangleWithAllThreeSides = (side1, side2, side3) => {
  validateNumericParam(side1, 'side1')
  validateNumericParam(side2, 'side2')
  validateNumericParam(side3, 'side3')
  if (
    side1 + side2 <= side3 ||
    side1 + side3 <= side2 ||
    side2 + side3 <= side1
  ) {
    throw new TypeError('Invalid Triangle sides.')
  }
  // Finding Semi perimeter of the triangle using formula
  const semi = (side1 + side2 + side3) / 2

  // Calculating the area of the triangle
  const area = Math.sqrt(
    semi * (semi - side1) * (semi - side2) * (semi - side3)
  )
  return Number(area.toFixed(2))
}

/**
 * @function areaParallelogram
 * @description Calculate the area of a parallelogram.
 * @param {Integer} base - Integer
 * @param {Integer} height - Integer
 * @return {Integer} - base * height
 * @see [areaParallelogram](https://en.wikipedia.org/wiki/Area#Dissection,_parallelograms,_and_triangles)
 * @example areaParallelogram(5, 6) = 24
 */
const areaParallelogram = (base, height) => {
  validateNumericParam(base, 'Base')
  validateNumericParam(height, 'Height')
  return base * height
}

/**
 * @function areaTrapezium
 * @description Calculate the area of a trapezium.
 * @param {Integer} base1 - Integer
 * @param {Integer} base2 - Integer
 * @param {Integer} height - Integer
 * @return {Integer} - (1 / 2) * (base1 + base2) * height
 * @see [areaTrapezium](https://en.wikipedia.org/wiki/Trapezoid)
 * @example areaTrapezium(5, 12, 10) = 85
 */
const areaTrapezium = (base1, base2, height) => {
  validateNumericParam(base1, 'Base One')
  validateNumericParam(base2, 'Base Two')
  validateNumericParam(height, 'Height')
  return (1 / 2) * (base1 + base2) * height
}

/**
 * @function areaCircle
 * @description Calculate the area of a circle.
 * @param {Integer} radius - Integer
 * @return {Integer} - Math.PI * radius ** 2
 * @see [areaCircle](https://en.wikipedia.org/wiki/Area_of_a_circle)
 * @example areaCircle(5, 12, 10) = 85
 */
const areaCircle = (radius) => {
  validateNumericParam(radius, 'Radius')
  return Math.PI * radius ** 2
}

/**
 * @function areaRhombus
 * @description Calculate the area of a rhombus.
 * @param {Integer} diagonal1 - Integer
 * @param {Integer} diagonal2 - Integer
 * @return {Integer} - (1 / 2) * diagonal1 * diagonal2
 * @see [areaRhombus](https://en.wikipedia.org/wiki/Rhombus)
 * @example areaRhombus(12, 10) = 60
 */
const areaRhombus = (diagonal1, diagonal2) => {
  validateNumericParam(diagonal1, 'diagonal one')
  validateNumericParam(diagonal2, 'diagonal two')
  return (1 / 2) * diagonal1 * diagonal2
}

const validateNumericParam = (param, paramName = 'param') => {
  if (typeof param !== 'number') {
    throw new TypeError('The ' + paramName + ' should be type Number')
  } else if (param < 0) {
    throw new Error('The ' + paramName + ' only accepts non-negative values')
  }
}

export {
  surfaceAreaCube,
  surfaceAreaSphere,
  areaRectangle,
  areaSquare,
  areaTriangle,
  areaParallelogram,
  areaTrapezium,
  areaCircle,
  areaRhombus,
  areaTriangleWithAllThreeSides
}
    `,
    java: `
    package com.thealgorithms.maths;

/**
 * Find the area of various geometric shapes
 */
public class Area {

    /**
     * String of IllegalArgumentException for radius
     */
    private static final String POSITIVE_RADIUS = "Must be a positive radius";

    /**
     * String of IllegalArgumentException for height
     */
    private static final String POSITIVE_HEIGHT = "Must be a positive height";

    /**
     * String of IllegalArgumentException for base
     */
    private static final String POSITIVE_BASE = "Must be a positive base";

    /**
     * Calculate the surface area of a cube.
     *
     * @param sideLength side length of cube
     * @return surface area of given cube
     */
    public static double surfaceAreaCube(final double sideLength) {
        if (sideLength <= 0) {
            throw new IllegalArgumentException("Must be a positive sideLength");
        }
        return 6 * sideLength * sideLength;
    }

    /**
     * Calculate the surface area of a sphere.
     *
     * @param radius radius of sphere
     * @return surface area of given sphere
     */
    public static double surfaceAreaSphere(final double radius) {
        if (radius <= 0) {
            throw new IllegalArgumentException(POSITIVE_RADIUS);
        }
        return 4 * Math.PI * radius * radius;
    }

    /**
     * Calculate the area of a rectangle.
     *
     * @param length length of a rectangle
     * @param width  width of a rectangle
     * @return area of given rectangle
     */
    public static double surfaceAreaRectangle(final double length, final double width) {
        if (length <= 0) {
            throw new IllegalArgumentException("Must be a positive length");
        }
        if (width <= 0) {
            throw new IllegalArgumentException("Must be a positive width");
        }
        return length * width;
    }

    /**
     * Calculate surface area of a cylinder.
     *
     * @param radius radius of the floor
     * @param height height of the cylinder.
     * @return volume of given cylinder
     */
    public static double surfaceAreaCylinder(final double radius, final double height) {
        if (radius <= 0) {
            throw new IllegalArgumentException(POSITIVE_RADIUS);
        }
        if (height <= 0) {
            throw new IllegalArgumentException(POSITIVE_RADIUS);
        }
        return 2 * (Math.PI * radius * radius + Math.PI * radius * height);
    }

    /**
     * Calculate the area of a square.
     *
     * @param sideLength side length of square
     * @return area of given square
     */
    public static double surfaceAreaSquare(final double sideLength) {
        if (sideLength <= 0) {
            throw new IllegalArgumentException("Must be a positive sideLength");
        }
        return sideLength * sideLength;
    }

    /**
     * Calculate the area of a triangle.
     *
     * @param base   base of triangle
     * @param height height of triangle
     * @return area of given triangle
     */
    public static double surfaceAreaTriangle(final double base, final double height) {
        if (base <= 0) {
            throw new IllegalArgumentException(POSITIVE_BASE);
        }
        if (height <= 0) {
            throw new IllegalArgumentException(POSITIVE_HEIGHT);
        }
        return base * height / 2;
    }

    /**
     * Calculate the area of a parallelogram.
     *
     * @param base   base of a parallelogram
     * @param height height of a parallelogram
     * @return area of given parallelogram
     */
    public static double surfaceAreaParallelogram(final double base, final double height) {
        if (base <= 0) {
            throw new IllegalArgumentException(POSITIVE_BASE);
        }
        if (height <= 0) {
            throw new IllegalArgumentException(POSITIVE_HEIGHT);
        }
        return base * height;
    }

    /**
     * Calculate the area of a trapezium.
     *
     * @param base1  upper base of trapezium
     * @param base2  bottom base of trapezium
     * @param height height of trapezium
     * @return area of given trapezium
     */
    public static double surfaceAreaTrapezium(final double base1, final double base2, final double height) {
        if (base1 <= 0) {
            throw new IllegalArgumentException(POSITIVE_BASE + 1);
        }
        if (base2 <= 0) {
            throw new IllegalArgumentException(POSITIVE_BASE + 2);
        }
        if (height <= 0) {
            throw new IllegalArgumentException(POSITIVE_HEIGHT);
        }
        return (base1 + base2) * height / 2;
    }

    /**
     * Calculate the area of a circle.
     *
     * @param radius radius of circle
     * @return area of given circle
     */
    public static double surfaceAreaCircle(final double radius) {
        if (radius <= 0) {
            throw new IllegalArgumentException(POSITIVE_RADIUS);
        }
        return Math.PI * radius * radius;
    }

    /**
     * Calculate the surface area of a hemisphere.
     *
     * @param radius radius of hemisphere
     * @return surface area of given hemisphere
     */
    public static double surfaceAreaHemisphere(final double radius) {
        if (radius <= 0) {
            throw new IllegalArgumentException(POSITIVE_RADIUS);
        }
        return 3 * Math.PI * radius * radius;
    }

    /**
     * Calculate the surface area of a cone.
     *
     * @param radius radius of cone.
     * @param height of cone.
     * @return surface area of given cone.
     */
    public static double surfaceAreaCone(final double radius, final double height) {
        if (radius <= 0) {
            throw new IllegalArgumentException(POSITIVE_RADIUS);
        }
        if (height <= 0) {
            throw new IllegalArgumentException(POSITIVE_HEIGHT);
        }
        return Math.PI * radius * (radius + Math.pow(height * height + radius * radius, 0.5));
    }
}
    `,
  },
  {
    id: "14",
    type: "mathematics",
    name: "arc length",
    python: `
    from math import pi


def arc_length(angle: int, radius: int) -> float:
    """
    >>> arc_length(45, 5)
    3.9269908169872414
    >>> arc_length(120, 15)
    31.415926535897928
    >>> arc_length(90, 10)
    15.707963267948966
    """
    return 2 * pi * radius * (angle / 360)


if __name__ == "__main__":
    print(arc_length(90, 10))
    `,
    js: "js",
    java: "java",
  },
  {
    id: "15",
    type: "mathematics",
    name: "area under curve",
    python: `
    """
Approximates the area under the curve using the trapezoidal rule
"""
from __future__ import annotations

from collections.abc import Callable


def trapezoidal_area(
    fnc: Callable[[float], float],
    x_start: float,
    x_end: float,
    steps: int = 100,
) -> float:
    """
    Treats curve as a collection of linear lines and sums the area of the
    trapezium shape they form
    :param fnc: a function which defines a curve
    :param x_start: left end point to indicate the start of line segment
    :param x_end: right end point to indicate end of line segment
    :param steps: an accuracy gauge; more steps increases the accuracy
    :return: a float representing the length of the curve

    >>> def f(x):
    ...    return 5
    >>> f"{trapezoidal_area(f, 12.0, 14.0, 1000):.3f}"
    '10.000'
    >>> def f(x):
    ...    return 9*x**2
    >>> f"{trapezoidal_area(f, -4.0, 0, 10000):.4f}"
    '192.0000'
    >>> f"{trapezoidal_area(f, -4.0, 4.0, 10000):.4f}"
    '384.0000'
    """
    x1 = x_start
    fx1 = fnc(x_start)
    area = 0.0
    for _ in range(steps):
        # Approximates small segments of curve as linear and solve
        # for trapezoidal area
        x2 = (x_end - x_start) / steps + x1
        fx2 = fnc(x2)
        area += abs(fx2 + fx1) * (x2 - x1) / 2
        # Increment step
        x1 = x2
        fx1 = fx2
    return area


if __name__ == "__main__":

    def f(x):
        return x**3 + x**2

    print("f(x) = x^3 + x^2")
    print("The area between the curve, x = -5, x = 5 and the x axis is:")
    i = 10
    while i <= 100000:
        print(f"with {i} steps: {trapezoidal_area(f, -5, 5, i)}")
        i *= 10
    `,
    js: "js",
    java: "java",
  },
  {
    id: "16",
    type: "dynamic programming",
    name: "all construct",
    python: `
    """
Program to list all the ways a target string can be
constructed from the given list of substrings
"""
from __future__ import annotations


def all_construct(target: str, word_bank: list[str] | None = None) -> list[list[str]]:
    """
        returns the list containing all the possible
        combinations a string(target) can be constructed from
        the given list of substrings(word_bank)
    >>> all_construct("hello", ["he", "l", "o"])
    [['he', 'l', 'l', 'o']]
    >>> all_construct("purple",["purp","p","ur","le","purpl"])
    [['purp', 'le'], ['p', 'ur', 'p', 'le']]
    """

    word_bank = word_bank or []
    # create a table
    table_size: int = len(target) + 1

    table: list[list[list[str]]] = []
    for _ in range(table_size):
        table.append([])
    # seed value
    table[0] = [[]]  # because empty string has empty combination

    # iterate through the indices
    for i in range(table_size):
        # condition
        if table[i] != []:
            for word in word_bank:
                # slice condition
                if target[i : i + len(word)] == word:
                    new_combinations: list[list[str]] = [
                        [word, *way] for way in table[i]
                    ]
                    # adds the word to every combination the current position holds
                    # now,push that combination to the table[i+len(word)]
                    table[i + len(word)] += new_combinations

    # combinations are in reverse order so reverse for better output
    for combination in table[len(target)]:
        combination.reverse()

    return table[len(target)]


if __name__ == "__main__":
    print(all_construct("jwajalapa", ["jwa", "j", "w", "a", "la", "lapa"]))
    print(all_construct("rajamati", ["s", "raj", "amat", "raja", "ma", "i", "t"]))
    print(
        all_construct(
            "hexagonosaurus",
            ["h", "ex", "hex", "ag", "ago", "ru", "auru", "rus", "go", "no", "o", "s"],
        )
    )
    `,
    js: "js",
    java: "java",
  },
  {
    id: "17",
    type: "dynamic programming",
    name: "abbreviation",
    python: `
    """
https://www.hackerrank.com/challenges/abbr/problem
You can perform the following operation on some string, :

1. Capitalize zero or more of 's lowercase letters at some index i
   (i.e., make them uppercase).
2. Delete all of the remaining lowercase letters in .

Example:
a=daBcd and b="ABC"
daBcd -> capitalize a and c(dABCd) -> remove d (ABC)
"""


def abbr(a: str, b: str) -> bool:
    """
    >>> abbr("daBcd", "ABC")
    True
    >>> abbr("dBcd", "ABC")
    False
    """
    n = len(a)
    m = len(b)
    dp = [[False for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = True
    for i in range(n):
        for j in range(m + 1):
            if dp[i][j]:
                if j < m and a[i].upper() == b[j]:
                    dp[i + 1][j + 1] = True
                if a[i].islower():
                    dp[i + 1][j] = True
    return dp[n][m]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    `,
    js: "js",
    java: "java",
  },
  {
    id: "18",
    type: "dynamic programming",
    name: "climbing stairs",
    python: `
    def climb_stairs(number_of_steps: int) -> int:
    """
    LeetCdoe No.70: Climbing Stairs
    Distinct ways to climb a number_of_steps staircase where each time you can either
    climb 1 or 2 steps.

    Args:
        number_of_steps: number of steps on the staircase

    Returns:
        Distinct ways to climb a number_of_steps staircase

    Raises:
        AssertionError: number_of_steps not positive integer

    >>> climb_stairs(3)
    3
    >>> climb_stairs(1)
    1
    >>> climb_stairs(-7)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    AssertionError: number_of_steps needs to be positive integer, your input -7
    """
    assert (
        isinstance(number_of_steps, int) and number_of_steps > 0
    ), f"number_of_steps needs to be positive integer, your input {number_of_steps}"
    if number_of_steps == 1:
        return 1
    previous, current = 1, 1
    for _ in range(number_of_steps - 1):
        current, previous = current + previous, current
    return current


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    `,
    js: `
    /**
 * @function ClimbStairs
 * @description You are climbing a stair case. It takes n steps to reach to the top.Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
 * @param {Integer} n - The input integer
 * @return {Integer} distinct ways can you climb to the top.
 * @see [Climb_Stairs](https://www.geeksforgeeks.org/count-ways-reach-nth-stair/)
 */

const climbStairs = (n) => {
  let prev = 0
  let cur = 1
  let temp

  for (let i = 0; i < n; i++) {
    temp = prev
    prev = cur
    cur += temp
  }
  return cur
}

export { climbStairs }

    `,
    java: `
    package com.thealgorithms.dynamicprogramming;

/* A DynamicProgramming solution for Climbing Stairs' problem Returns the
    distinct ways can you climb to the staircase by either climbing 1 or 2 steps.

    Link : https://medium.com/analytics-vidhya/leetcode-q70-climbing-stairs-easy-444a4aae54e8
*/
public class ClimbingStairs {

    public static int numberOfWays(int n) {

        if (n == 1 || n == 0) {
            return n;
        }
        int prev = 1;
        int curr = 1;

        int next;

        for (int i = 2; i <= n; i++) {
            next = curr + prev;
            prev = curr;

            curr = next;
        }

        return curr;
    }
}

    `,
  },
  {
    id: "19",
    type: "dynamic programming",
    name: "fast fibonacci",
    python: `
    """
This program calculates the nth Fibonacci number in O(log(n)).
It's possible to calculate F(1_000_000) in less than a second.
"""
from __future__ import annotations

import sys


def fibonacci(n: int) -> int:
    """
    return F(n)
    >>> [fibonacci(i) for i in range(13)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    """
    if n < 0:
        raise ValueError("Negative arguments are not supported")
    return _fib(n)[0]


# returns (F(n), F(n-1))
def _fib(n: int) -> tuple[int, int]:
    if n == 0:  # (F(0), F(1))
        return (0, 1)

    # F(2n) = F(n)[2F(n+1) − F(n)]
    # F(2n+1) = F(n+1)^2+F(n)^2
    a, b = _fib(n // 2)
    c = a * (b * 2 - a)
    d = a * a + b * b
    return (d, c + d) if n % 2 else (c, d)


if __name__ == "__main__":
    n = int(sys.argv[1])
    print(f"fibonacci({n}) is {fibonacci(n)}")
    `,
    js: "js",
    java: "java",
  },
];
const lanData = [
  { id: 0, icon: "fa-brands fa-python", lan: "Python", userNum: "15.7M" },
  { id: 1, icon: "fa-brands fa-java", lan: "Java", userNum: "9.4M" },
  {
    id: 2,
    icon: "fa-brands fa-square-js",
    lan: "JavaScript",
    userNum: "13.8M",
  },
  { id: 3, icon: "fa-brands fa-rust", lan: "Rust", userNum: "2.8M" },
  { id: 4, icon: "fa-brands fa-golang", lan: "Go", userNum: "2.1M" },
  { id: 5, icon: "fa-brands fa-php", lan: "PHP", userNum: "6.3M" },
  { id: 6, icon: "fa-brands fa-swift", lan: "Swift", userNum: "2.1M" },
];

export default function App() {
  const [searchText, setSearchText] = useState("");

  return (
    <div className="app">
      <Header searchText={searchText} setSearchText={setSearchText} />
      <Main searchText={searchText} />
      <Footer />
    </div>
  );
}

// HEADER
function Header({ searchText, setSearchText }) {
  return (
    <div className="header">
      <div className="header-container">
        <HeaderLogo />
        <HeaderSearch searchText={searchText} setSearchText={setSearchText} />
      </div>
    </div>
  );
}
function HeaderLogo() {
  return (
    <div className="header-logo">
      <span>*Khwarizmi-Legacy*</span>
    </div>
  );
}
function HeaderSearch({ searchText, setSearchText }) {
  const [searchActive, setSearchActive] = useState(false);

  function handleIcon() {
    setSearchActive(!searchActive);
  }

  function handleClear() {
    setSearchText("");
  }

  return (
    <form className="header-btns">
      <div className={`header-search ${searchActive ? "active" : null}`}>
        <div className="icon" onClick={handleIcon}></div>
        <div className="input">
          <input
            type="text"
            placeholder="Search"
            id="mysearch"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
        </div>
        <span className="clear" onClick={handleClear}></span>
      </div>
    </form>
  );
}

// MAIN
function Main({ searchText }) {
  const [get, setGet] = useState(false);
  const [curAlId, setCurAlId] = useState(0);

  function handleGet(id) {
    setGet(true);
    setCurAlId(id);
  }
  // console.log(curAlId);
  return (
    <div className="main">
      {/* Search Results */}
      {searchText && (
        <SearchResult onGet={handleGet}>
          {data
            .filter((al) => al.name === searchText)
            .map((al) => (
              <AlgorithmCard
                type={al.type}
                algorithmName={al.name}
                onGet={() => handleGet(al.id)}
                key={al.id}
              />
            ))}
        </SearchResult>
      )}

      {/* Display Code */}
      {get && (
        <DisplayCode
          title={data.at(curAlId).name}
          python={data.at(curAlId).python}
          js={data.at(curAlId).js}
          java={data.at(curAlId).java}
          onGet={setGet}
        />
      )}

      {/* Top Algorithms */}
      <SubTitle title="top algorithms" />
      <TopAlgorithms>
        {data.slice(0, 4).map((al) => (
          <AlgorithmCard
            type={al.type}
            algorithmName={al.name}
            onGet={() => handleGet(al.id)}
            key={al.id}
          />
        ))}
      </TopAlgorithms>

      {/* About */}
      <About />

      {/* Programming Languages */}
      <ProgrammingLan />
    </div>
  );
}
function SearchResult({ onGet, children }) {
  return (
    <div className="main-search-result">
      <SubTitle title="Search result" />
      <div className="main-srch-container">{children}</div>
    </div>
  );
}
function DisplayCode({ title, python, java, js, onGet }) {
  const [curLan, setCurLan] = useState("python");

  return (
    <div className="main-display-code">
      <div className="display-title">
        <h3>{title}</h3>
        <div className="x" onClick={() => onGet(false)}>
          ✖
        </div>
      </div>
      <div className="display-code multiline">
        {curLan === "python" ? python : curLan === "java" ? java : js}
      </div>
      <div className="display-icons">
        <i
          onClick={() => setCurLan("python")}
          className="fa-brands fa-python button"
        ></i>
        <i
          onClick={() => setCurLan("js")}
          className="fa-brands fa-js button"
        ></i>
        <i
          onClick={() => setCurLan("java")}
          className="fa-brands fa-java button"
        ></i>
      </div>
    </div>
  );
}
function TopAlgorithms({ children }) {
  return (
    <div className="best-algorithms">
      <div className="algorithm-container">{children}</div>
    </div>
  );
}
function About() {
  return (
    <div className="main-about-us">
      <div className="al-description">
        <h3>what is an algorithm?</h3>
        <p>
          An algorithm is a set of rules that takes in one or more inputs, then
          performs inner calculations and data manipulations and returns an
          output or a set of outputs. In short, algorithms make life easy. From
          complex data manipulations and hashes, to simple arithmetic,
          algorithms follow a set of steps to produce a useful result. One
          example of an algorithm would be a simple function that takes two
          input values, adds them together, and returns their sum.
        </p>
      </div>
    </div>
  );
}
function ProgrammingLan() {
  return (
    <div className="programming-languages">
      <SubTitle title="Programming Languages" />
      <div className="p-lan-descriptions">
        <p>
          A programming language is a way for programmers (developers) to
          communicate with computers. Programming languages consist of a set of
          rules that allows string values to be converted into various ways of
          generating machine code, or, in the case of visual programming
          languages, graphical elements. <br />
          <br />
          Here are some top programming languages:
        </p>
      </div>
      <div className="p-lan-card">
        {lanData.map((lan) => (
          <AboutLan
            icon={lan.icon}
            lan={lan.lan}
            userNum={lan.userNum}
            key={lan.id}
          />
        ))}
      </div>
    </div>
  );
}
function AboutLan({ icon, lan, userNum }) {
  return (
    <div className="about-lan">
      <div className="aboutlan-icon">
        <i className={icon}></i>
      </div>
      <div className="aboutlan-icon">{lan}</div>
      <div className="aboutlan-usernum">
        <i class="fa-solid fa-star"></i>
        <span>{userNum}</span>
      </div>
    </div>
  );
}
function SubTitle({ title }) {
  return <h2 className="subtitle">{title}</h2>;
}
function AlgorithmCard({ type, algorithmName, onGet }) {
  return (
    <div className="algorithm-card">
      <p className="al-type">{type}</p>
      <h3>{algorithmName}</h3>
      <div className="al-icons">
        <div className="lan-icons">
          <i className="fa-brands fa-python button"></i>
          <i className="fa-brands fa-js button"></i>
          <i className="fa-brands fa-java button"></i>
        </div>
        <div className="get-button" onClick={onGet}>
          GET
        </div>
      </div>
    </div>
  );
}

// FOOTER
function Footer() {
  return <div className="footer">*Khwarizmi-Legacy*</div>;
}
