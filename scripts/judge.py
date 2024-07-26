import subprocess
import sys
import math

'''
Runs a program, providing input file through stdin pipe
and capturing stdout pipe to judge output of program.
'''
def run_program(program_path, input_data, *args):
  if args:
    popen_args = [program_path, *args]
  else:
    popen_args = [program_path]
  process = subprocess.Popen(popen_args,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)

  data = open(input_data).read()
  try:
    process.stdin.write(data)
  except:
     pass
  output, _ = process.communicate()

  return output


def score(mine, left, right):
    gold = "gold"
    element_cnt = [0] * len(gold)
    gold_cnt = 0
    rock_cnt = 0

    i = left
    while i <= right:
        if mine[i] in gold:
            element_cnt[gold.index(mine[i])] += 1

            if (right - i) >= 3:
                #enough space left for "gold" or "dlog"
                if (mine[i:i + 4] == "gold") or (mine[i:i + 4] == "dlog"):
                    gold_cnt += 1
                    for c in mine[i + 1:i + 4]:
                        element_cnt[gold.index(c)] += 1
                    i += 3  # skip past other 3 elements, too
        else:
            rock_cnt += 1
        
        i += 1  # always go one ahead at least
     
    def scalar(x):
        return 1 + (1200.0) * math.exp(-x / 20000.0) / 100.0

    width = right - left + 1
    return math.floor(((4 * gold_cnt) + (4 * min(element_cnt) - rock_cnt)) * scalar(width))


'''
NOTE!!!!!!!!!!!
This judging program is assuming exactly 100,000 chars in the mine
This final mine data file was generated with the following command:
    python3 gen_golddigger.py -s 100000 -r 0.25 -l 50
'''
def judge(output):
  '''
  Process student output here and return score.
  '''
  
  '''
  First try to read their potential 10 int positions (5 pairs of left:right)
  '''
  integers = []
  nums = output.split()
  for n in range(10):
    try:
      number = int(nums[n])
      integers.append(number)
    except:
      return -1  # not an integer, not enough integers, or student code crashed

  # how many pairs of positions are there
  pair_cnt = len(integers) // 2
  zones = []
  for p in range(pair_cnt):
    pA = p * 2
    pB = (p * 2) + 1
    if integers[pA] >= 0 and integers[pA] < 100000 and integers[pB] >= 0 and integers[pB] < 100000:
      # each position of the pair is in the mine, at least

      if integers[pA] <= integers[pB]:
        # the left position is at least left or the same of/as the right

        zones.append([integers[pA], integers[pB]])
      else:
         return -3  # left:right of a pair are out of order
    else:
       return -2  # a position is outside the range of the mine
  pair_cnt = len(zones)

  '''
  At this point, the zones are valid left:right pairs in the range of the mine.
  Check if they overlap/collide.
  '''
  p = pair_cnt - 1
  while p > 0:
    # check all zones earlier than zone[p] for overlap
    for i in range(p):
      collision = True
      if zones[i][1] < zones[p][0]:
        collision = False
      if zones[p][1] < zones[i][0]:
        collision = False
      
      if collision: 
        return -4  # two zones overlap
    p -= 1

  '''
  If made it here, the 5 zones are all good, so let's find the actual score!
  '''
  mine = []
  with open(sys.argv[1], "r") as f:
    for c in f.read():
      if c in "gold-":
          mine.append(c)

  total = 0
  for z in zones:
     total += score(mine, z[0], z[1])

  return total


if __name__ == "__main__":
  import sys

  if len(sys.argv) < 3:
    print("Usage:", sys.argv[0], "<input_data_file> <program_path> <program_args>")
    sys.exit(1)

  input_data = sys.argv[1]
  program_path = sys.argv[2]
  program_args = sys.argv[3:]

  if len(program_args) > 0:
    output = run_program(program_path, input_data, *program_args)
  else:
    output = run_program(program_path, input_data)

  # Process the captured output here and print the associated score
  print(int(judge(output)))
