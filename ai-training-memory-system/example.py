#!/usr/bin/env python3
"""
Simple example demonstrating the AI Training Memory System.
Run this script to see the system in action.
"""

from src.core.memory import AITrainingMemory


def main():
    print("="*70)
    print("AI TRAINING MEMORY SYSTEM - SIMPLE EXAMPLE")
    print("="*70)
    print()
    
    # Initialize memory system
    print("📚 Initializing memory system...")
    memory = AITrainingMemory()
    print("✅ System initialized!\n")
    
    # Example 1: First query
    print("="*70)
    print("Example 1: Solving a Problem")
    print("="*70)
    problem1 = "What is 5 plus 3?"
    print(f"Problem: {problem1}")
    result1 = memory.solve_problem(problem1)
    print(f"Answer: {result1['answer']}")
    print(f"Method: {result1['method']}")
    print(f"Time: {result1['time']:.4f}s\n")
    
    # Example 2: Duplicate query (instant retrieval)
    print("="*70)
    print("Example 2: Duplicate Detection (Instant Retrieval)")
    print("="*70)
    print(f"Problem: {problem1} (same as before)")
    result2 = memory.solve_problem(problem1)
    print(f"Answer: {result2['answer']}")
    print(f"Method: {result2['method']} ⚡")
    print(f"Time: {result2['time']:.4f}s")
    print(f"Speed improvement: {result1['time']/result2['time']:.1f}x faster!\n")
    
    # Example 3: Different operations
    print("="*70)
    print("Example 3: Different Mathematical Operations")
    print("="*70)
    problems = [
        "Calculate 10 minus 4",
        "What is 7 times 6?",
        "Divide 20 by 4"
    ]
    
    for problem in problems:
        result = memory.solve_problem(problem)
        print(f"{problem} = {result['answer']}")
    print()
    
    # Example 4: Semantic similarity
    print("="*70)
    print("Example 4: Semantic Similarity (Different Wordings)")
    print("="*70)
    similar_problems = [
        "Add 5 and 3",
        "Calculate 5 + 3",
        "What's the sum of 5 and 3?"
    ]
    
    for problem in similar_problems:
        result = memory.solve_problem(problem)
        print(f"{problem}")
        print(f"  Answer: {result['answer']}, Method: {result['method']}\n")
    
    # Example 5: Training
    print("="*70)
    print("Example 5: Training the System")
    print("="*70)
    print("Adding training examples...")
    
    training_examples = [
        {'problem': 'What is 12 times 12?', 'answer': 144, 'operation': 'multiplication'},
        {'problem': 'Calculate 100 divided by 5', 'answer': 20, 'operation': 'division'},
    ]
    
    memory.training_phase(training_examples)
    print(f"✅ Training complete! Cycles: {memory.training_cycles}\n")
    
    # Example 6: Memory statistics
    print("="*70)
    print("Example 6: Memory Statistics")
    print("="*70)
    memory.show_memory()
    
    # Example 7: Performance metrics
    print("="*70)
    print("Example 7: Performance Summary")
    print("="*70)
    report = memory.performance_tracker.generate_report()
    print(f"Total Queries: {report['summary']['total_queries']}")
    print(f"Memory Hits: {report['summary']['memory_hits']}")
    print(f"Hit Rate: {report['summary']['hit_rate']}")
    print(f"Average Solve Time: {report['timing']['avg_solve_time']}")
    print()
    
    print("="*70)
    print("🎉 Demo Complete!")
    print("="*70)
    print("\nTry the CLI interface: python -m src.interface.cli")
    print("Or run full demos: python -m src.interface.demo")
    print()


if __name__ == "__main__":
    main()
