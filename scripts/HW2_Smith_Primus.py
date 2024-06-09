def sum_of_digits(number):
    return sum(int(digit) for digit in str(abs(number)))

result = sum_of_digits(100)
print(result)   # Output: 6

def decimal_to_base(number, base):
    if not (2 <= base <= 10):
        raise ValueError("Base must be between 2 and 9")
    if number == 0:
        return '0'
    digits = ''
    while number > 0:
        digits += str(number % base)
        number //= base
    return digits[::-1]

print(decimal_to_base(100, 2))  # Output: 1010 (binary)
print(decimal_to_base(100, 3))  # Output: 101 (base 3)
print(decimal_to_base(1, 4))  # Output: 22 (base 4)
print(decimal_to_base(100, 5))  # Output: 20 (base 5)
print(decimal_to_base(100, 6))  # Output: 14 (base 6)
print(decimal_to_base(100, 7))  # Output: 13 (base 7)
print(decimal_to_base(100, 8))  # Output: 12 (base 8)
print(decimal_to_base(100, 9))  # Output: 11 (base 9)
print(decimal_to_base(100, 10))  # Output: 11 (base 9)


def has_equal_digit_counts(number_str, base):
    # Convert the string to a list of integers
    digits = [int(digit) for digit in number_str]
    
    # Check if all digits are within the range for the given base
    if not all(0 <= digit < base for digit in digits):
        raise ValueError("All digits must be within the range allowed by the base")
    
    # Count the occurrences of each digit
    digit_counts = {digit: digits.count(digit) for digit in range(base)}
    
    # Get the set of counts to check if all are equal
    counts_set = set(digit_counts.values())
    
    # If there's only one unique count and it's not zero, return True
    return len(counts_set) == 1 and next(iter(counts_set)) != 0

test_numbers = {
    2: ['000100111', '01', '000000111'],
    3: ['00001122112', '012', '121212120'],
    4: ['30330121201', '0123', '3333003210323'],
    5: ['442100231', '01234', '43204314404132'],
    6: ['04321501234', '012345', '1542230331520441'],
    7: ['0123421065345', '0123456', '65142435016161520042516'],
    8: ['016543270123456', '01234567', '12345006776712062341526'],
    9: ['01234562108734567', '012345678', '1820736210000281028102']
}

# Running the tests
for base, numbers in test_numbers.items():
    for number in numbers:
        result = has_equal_digit_counts(number, base)
        print(f"Base {base}, Number {number}: {result}")

def process_number(number, base):
    # Convert the original number to the given base and store it
    # Initialize the final number with the original number in base
    final_number = str(number)
    current_number = number
    num_string = str(number)
    num_string += "->"
    
    while True:
        # Find the sum of digits of the current number
        #print("current_number")
        #print(current_number)
        sum_digits = sum_of_digits(current_number)
    
        
        temp = int(decimal_to_base(sum_digits, base))
        
        # If the sum is a single digit, break the loop
        if temp < 10 and temp >= 0:
            sum_in_base = decimal_to_base(sum_digits, base)
            final_number += sum_in_base
            num_string += sum_in_base
            break
        
        # Convert the sum back to the base
        sum_in_base = decimal_to_base(sum_digits, base)
        #print("sum_in_base")
        #print(sum_in_base)
        
        # Append the sum in base to the final number
        final_number += sum_in_base
        num_string += sum_in_base
        num_string += "->"
        
        # Update the current number with the sum for the next iteration
        current_number = int(sum_in_base)
    
    return final_number, num_string



base = 2 # Replace with the desired base
number = 10010  # Replace with the initial number
num_string = ""
result, num_string = process_number(number, base)
print(result)
print(num_string)


has_equal_digit_counts(result, base)


def find_balanced_numbers(base):
    count = 0
    number = 0
    balanced_numbers = []

    while count < 10:
        number_in_base = decimal_to_base(number, base)
        fin_num, num_string = process_number(int(number_in_base), base)
        if has_equal_digit_counts(fin_num, base):
            balanced_numbers.append(number_in_base)
            print(num_string)
            count += 1
        number += 1
    
    return balanced_numbers


for base in range(2, 11):
    print("Base ", base, ":")
    balanced_numbers = find_balanced_numbers(base)
    #print(f"Base {base} balanced numbers: {balanced_numbers}")




