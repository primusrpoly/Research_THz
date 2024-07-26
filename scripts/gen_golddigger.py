#!/usr/bin/env python3

import sys
import argparse
import random

DEBUG = True

parser = argparse.ArgumentParser(
    description = 'Generates a random gold mine given its parameters.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument('-s', '--size', type = int, default = 2000, help = 'Integer number of characheters in the mine > 0')
parser.add_argument('-r', '--rock', type = float, default = 0.1, help = 'precentage (0...1) of mine that is rock')
parser.add_argument('-l', '--largest', type = int, default = 20, help = 'Integer size of largest (or so) rock > 0')
parser.add_argument('-o', '--outfile', type = argparse.FileType('w'), default = 'mine.dat', help = 'Output file name')
args = parser.parse_args()

arg_errors = False

if args.size <= 0:
    print('\nsize must be > 0')
    arg_errors = True

if args.rock < 0 or args.rock > 1:
    print('\nrock percentage must be between [0...1]')
    arg_errors = True

if args.largest <= 0:
    print('\nlargest rock size must be > 0')
    arg_errors = True


if args.outfile.closed:
    print('\nThe file is closed for some reason')
    arg_errors = True

if arg_errors:
    print('')
    parser.print_help()
    sys.exit(1)

f = args.outfile
size = args.size
rock = args.rock
largest = args.largest

mine = ['x'] * size

def constrain(x, low, high):
	return max(low, min(x, high))


gold = list("gold")
i = random.choice(range(len(gold)))


# fill out the mine with gold elements

v = random.random()*4 - 2
for m in range(size):
    delta = (v >= 0.75) - (v <= -0.75)
    if(abs(v) > 1.8):
         delta = random.choice([-2, +2])
    #print(delta)
    i = (i + delta) % 4
    mine[m] = gold[i]

    v += random.random()*1.4 - 0.7
    v = constrain(v, -2, +2)

# add some rock

rock_cnt = 0
while (rock_cnt / size) < rock:
    # pick random rock left and right
    left = -1
    right = -1
    if(random.choice([True, False])):
       # left first
       left = random.randint(0, size - 1)
       right = constrain(left + random.randint(0, largest - 1), left, size - 1)
    else:
       # right first
       right = random.randint(0, size - 1)
       left = constrain(right - random.randint(0, largest - 1), 0, right)

    rock_cnt += right - left + 1
    i = left
    while i <= right:
        mine[i] = '-'
        i += 1



for m in range(size):
    f.write(mine[m])
