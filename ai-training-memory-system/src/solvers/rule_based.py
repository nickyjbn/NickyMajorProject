"""
Rule-based mathematical problem solver.
Extracts numbers and operations to solve problems.
"""
import re
from typing import List, Dict, Optional, Any


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text using regex.
    
    Args:
        text: Input text string
        
    Returns:
        List of extracted numbers as floats
    """
    # Pattern matches integers and decimals, including negative numbers
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers


def identify_operation(text: str) -> str:
    """
    Identify mathematical operation from keywords in text.
    
    Args:
        text: Input text string
        
    Returns:
        Operation name ('addition', 'subtraction', 'multiplication', 'division', 'unknown')
    """
    text_lower = text.lower()
    
    # Addition keywords
    if any(word in text_lower for word in ['add', 'plus', 'sum', 'total', 'combine', '+']):
        return 'addition'
    
    # Subtraction keywords
    if any(word in text_lower for word in ['subtract', 'minus', 'difference', 'less', 'remove', '-']):
        return 'subtraction'
    
    # Multiplication keywords
    if any(word in text_lower for word in ['multiply', 'times', 'product', 'of', '*', 'x']):
        return 'multiplication'
    
    # Division keywords
    if any(word in text_lower for word in ['divide', 'divided by', 'quotient', 'per', '/', 'split']):
        return 'division'
    
    return 'unknown'


def solve_addition(numbers: List[float]) -> float:
    """
    Sum all numbers.
    
    Args:
        numbers: List of numbers to add
        
    Returns:
        Sum of all numbers
    """
    return sum(numbers)


def solve_subtraction(numbers: List[float]) -> float:
    """
    Subtract subsequent numbers from the first.
    
    Args:
        numbers: List of numbers (first - rest)
        
    Returns:
        Result of subtraction
    """
    if len(numbers) == 0:
        return 0.0
    if len(numbers) == 1:
        return numbers[0]
    
    result = numbers[0]
    for num in numbers[1:]:
        result -= num
    
    return result


def solve_multiplication(numbers: List[float]) -> float:
    """
    Multiply all numbers together.
    
    Args:
        numbers: List of numbers to multiply
        
    Returns:
        Product of all numbers
    """
    if len(numbers) == 0:
        return 0.0
    
    result = 1.0
    for num in numbers:
        result *= num
    
    return result


def solve_division(numbers: List[float]) -> Optional[float]:
    """
    Divide first number by second.
    
    Args:
        numbers: List of numbers (first / second)
        
    Returns:
        Result of division or None if division by zero
    """
    if len(numbers) < 2:
        return None
    
    if numbers[1] == 0:
        return None  # Division by zero
    
    return numbers[0] / numbers[1]


def solve_with_rules(problem: str) -> Dict[str, Any]:
    """
    Main rule-based solving function.
    
    Args:
        problem: Mathematical problem as text
        
    Returns:
        Dictionary with solution details
    """
    # Extract numbers and operation
    numbers = extract_numbers(problem)
    operation = identify_operation(problem)
    
    # Initialize result
    result = {
        'numbers_extracted': numbers,
        'operation': operation,
        'answer': None,
        'explanation': '',
        'success': False
    }
    
    if len(numbers) == 0:
        result['explanation'] = 'No numbers found in problem'
        return result
    
    # Solve based on operation
    try:
        if operation == 'addition':
            answer = solve_addition(numbers)
            result['answer'] = answer
            result['explanation'] = f"Added {' + '.join(map(str, numbers))} = {answer}"
            result['success'] = True
        
        elif operation == 'subtraction':
            answer = solve_subtraction(numbers)
            result['answer'] = answer
            result['explanation'] = f"Subtracted {numbers[0]} - {' - '.join(map(str, numbers[1:]))} = {answer}"
            result['success'] = True
        
        elif operation == 'multiplication':
            answer = solve_multiplication(numbers)
            result['answer'] = answer
            result['explanation'] = f"Multiplied {' × '.join(map(str, numbers))} = {answer}"
            result['success'] = True
        
        elif operation == 'division':
            answer = solve_division(numbers)
            if answer is not None:
                result['answer'] = answer
                result['explanation'] = f"Divided {numbers[0]} ÷ {numbers[1]} = {answer}"
                result['success'] = True
            else:
                result['explanation'] = 'Division by zero error'
        
        else:
            result['explanation'] = f"Unknown operation, numbers found: {numbers}"
    
    except Exception as e:
        result['explanation'] = f"Error during calculation: {str(e)}"
    
    return result
