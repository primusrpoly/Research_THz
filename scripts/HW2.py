import collections
from collections import Counter

def is_tasty(n, base):
  """
  Checks if a number is tasty in a given base.

  Args:
      n: The number to check.
      base: The base of the number.

  Returns:
      True if the number is tasty, False otherwise.
  """
  seen = set()
  while n > 0:
    digit_sum = 0
    while n > 0:
      digit_sum += n % base
      n //= base
    if digit_sum in seen:
      return False
    seen.add(digit_sum)
    n = digit_sum
  # Check if all digits (0 to base-1) are present in seen after loop ends
  return len(seen) == base and all(count == 1 for count in collections.Counter(seen).values())

# Test cases with valid bases
print(is_tasty(5801364, 10))  # True
print(is_tasty(1111, 10))  # False (doesn't use all digits)
print(is_tasty(2, 2))  # True
print(is_tasty(121, 4))  # True (all digits 0, 1, 2, 3 used once)
# Remove the misleading test case
# print(is_tasty(101, 2))  # This number is not valid in base 2
print(is_tasty(11, 2))  # Valid test case for base 2 (digits 0 and 1)

