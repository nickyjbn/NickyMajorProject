"""
Demo scenarios showcasing the AI Training Memory System capabilities.
"""
from ..core.memory import AITrainingMemory
import time


class DemoScenarios:
    """Collection of demonstration scenarios."""
    
    def __init__(self):
        """Initialize demo with fresh memory system."""
        self.memory = AITrainingMemory()
    
    def demo_1_basic_memory(self):
        """
        Demo 1: Basic Memory - Duplicate detection and instant retrieval.
        Shows how the system remembers questions and provides instant answers.
        """
        print("\n" + "="*70)
        print("DEMO 1: BASIC MEMORY - DUPLICATE DETECTION")
        print("="*70)
        
        print("\n📝 Scenario: Ask the same question multiple times")
        print("-" * 70)
        
        problem = "What is 5 plus 3?"
        
        # First time asking
        print(f"\n1️⃣  First time asking: '{problem}'")
        result1 = self.memory.solve_problem(problem)
        print(f"   ⏱️  Time: {result1['time']:.4f}s")
        print(f"   ⚙️  Method: {result1['method']}")
        print(f"   ✨ Answer: {result1['answer']}")
        
        time.sleep(0.5)
        
        # Second time asking (should be instant)
        print(f"\n2️⃣  Second time asking: '{problem}'")
        result2 = self.memory.solve_problem(problem)
        print(f"   ⏱️  Time: {result2['time']:.4f}s (🚀 {result1['time']/result2['time']:.1f}x faster!)")
        print(f"   ⚙️  Method: {result2['method']}")
        print(f"   ✨ Answer: {result2['answer']}")
        
        time.sleep(0.5)
        
        # Third time asking
        print(f"\n3️⃣  Third time asking: '{problem}'")
        result3 = self.memory.solve_problem(problem)
        print(f"   ⏱️  Time: {result3['time']:.4f}s")
        print(f"   ⚙️  Method: {result3['method']}")
        print(f"   ✨ Answer: {result3['answer']}")
        
        print("\n✅ Key Insight: Questions asked before are retrieved instantly from memory!")
        print(f"   Memory hit rate: {(self.memory.memory_hit_count / self.memory.total_queries * 100):.1f}%")
        
        print("\n" + "="*70 + "\n")
    
    def demo_2_semantic_similarity(self):
        """
        Demo 2: Semantic Similarity - Understanding different wordings.
        Shows how the system recognizes similar questions with different phrasing.
        """
        print("\n" + "="*70)
        print("DEMO 2: SEMANTIC SIMILARITY - DIFFERENT WORDINGS")
        print("="*70)
        
        print("\n📝 Scenario: Ask similar questions with different wordings")
        print("-" * 70)
        
        problems = [
            "What is 10 plus 5?",
            "Calculate 10 + 5",
            "Add 10 and 5",
            "What's the sum of 10 and 5?"
        ]
        
        for i, problem in enumerate(problems, 1):
            print(f"\n{i}️⃣  Question: '{problem}'")
            result = self.memory.solve_problem(problem)
            print(f"   ✨ Answer: {result['answer']}")
            print(f"   ⚙️  Method: {result['method']}")
            print(f"   🎯 Confidence: {result['confidence']:.2f}")
            
            # Check for similar problems
            similar = self.memory.similarity_search(problem, k=3, threshold=0.7)
            if similar and len(similar) > 1:
                print(f"   🔗 Found {len(similar)-1} similar question(s) in memory")
            
            time.sleep(0.3)
        
        print("\n✅ Key Insight: System understands semantic similarity!")
        print("   Different wordings of same question are recognized as similar.")
        
        print("\n" + "="*70 + "\n")
    
    def demo_3_learning_from_mistakes(self):
        """
        Demo 3: Learning from Mistakes - Training and improvement.
        Shows how the system learns from training examples and improves.
        """
        print("\n" + "="*70)
        print("DEMO 3: LEARNING FROM MISTAKES - TRAINING MODE")
        print("="*70)
        
        print("\n📝 Scenario: System learns from training examples")
        print("-" * 70)
        
        # Before training
        print("\n📊 Before Training:")
        print(f"   Training cycles: {self.memory.training_cycles}")
        print(f"   Problems in memory: {len(self.memory.memory_documents)}")
        
        # Training phase
        print("\n🎓 Initiating training phase...")
        training_examples = [
            {'problem': 'What is 7 times 8?', 'answer': 56, 'operation': 'multiplication'},
            {'problem': 'Calculate 100 divided by 5', 'answer': 20, 'operation': 'division'},
            {'problem': 'Subtract 15 from 50', 'answer': 35, 'operation': 'subtraction'},
            {'problem': 'Add 23 and 17', 'answer': 40, 'operation': 'addition'},
            {'problem': 'What is 12 times 12?', 'answer': 144, 'operation': 'multiplication'},
        ]
        
        self.memory.training_phase(training_examples)
        print("✅ Training complete!")
        
        # After training
        print("\n📊 After Training:")
        print(f"   Training cycles: {self.memory.training_cycles}")
        print(f"   Problems in memory: {len(self.memory.memory_documents)}")
        
        # Test learned knowledge
        print("\n🧪 Testing learned knowledge:")
        test_problems = [
            "What is 7 times 8?",  # Exact match from training
            "Calculate 12 times 12",  # Similar to training
        ]
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n{i}️⃣  Test: '{problem}'")
            result = self.memory.solve_problem(problem)
            print(f"   ✨ Answer: {result['answer']}")
            print(f"   ⚙️  Method: {result['method']}")
            print(f"   ⏱️  Time: {result['time']:.4f}s")
        
        print("\n✅ Key Insight: System learns from training and retrieves faster!")
        
        print("\n" + "="*70 + "\n")
    
    def demo_4_complex_problem_solving(self):
        """
        Demo 4: Complex Problem Solving - Multi-step reasoning.
        Shows the system handling various mathematical operations.
        """
        print("\n" + "="*70)
        print("DEMO 4: COMPLEX PROBLEM SOLVING - MULTI-STEP REASONING")
        print("="*70)
        
        print("\n📝 Scenario: Solve various types of mathematical problems")
        print("-" * 70)
        
        problems = [
            "What is 25 plus 37?",
            "Calculate 100 minus 45",
            "What is 8 times 9?",
            "Divide 144 by 12",
            "Add 123 and 456 and 789",
        ]
        
        for i, problem in enumerate(problems, 1):
            print(f"\n{i}️⃣  Problem: '{problem}'")
            result = self.memory.solve_problem(problem)
            
            print(f"   ✨ Answer: {result['answer']}")
            print(f"   📋 Explanation: {result['explanation']}")
            print(f"   🔢 Numbers: {result.get('numbers_extracted', [])}")
            print(f"   ➕ Operation: {result.get('operation', 'unknown')}")
            print(f"   ⚙️  Method: {result['method']}")
            print(f"   🎯 Confidence: {result['confidence']:.2f}")
            print(f"   ⏱️  Time: {result['time']:.4f}s")
            
            time.sleep(0.3)
        
        print("\n✅ Key Insight: System handles multiple operations with rule-based solving!")
        print(f"   Total queries: {self.memory.total_queries}")
        print(f"   Average solve time: {sum(self.memory.solve_times)/len(self.memory.solve_times):.4f}s")
        
        print("\n" + "="*70 + "\n")
    
    def run_all_demos(self):
        """Run all demonstration scenarios in sequence."""
        print("\n" + "="*70)
        print("🎬 AI TRAINING MEMORY SYSTEM - COMPLETE DEMONSTRATION")
        print("="*70)
        print("\nThis demonstration will showcase 4 key capabilities:")
        print("  1. Basic Memory & Duplicate Detection")
        print("  2. Semantic Similarity Understanding")
        print("  3. Learning from Training Examples")
        print("  4. Complex Problem Solving")
        print("\n" + "="*70)
        
        input("\nPress Enter to start Demo 1...")
        self.demo_1_basic_memory()
        
        input("Press Enter to start Demo 2...")
        self.demo_2_semantic_similarity()
        
        input("Press Enter to start Demo 3...")
        self.demo_3_learning_from_mistakes()
        
        input("Press Enter to start Demo 4...")
        self.demo_4_complex_problem_solving()
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL SUMMARY")
        print("="*70)
        self.memory.show_memory()
        
        print("\n🎉 Demonstration Complete!")
        print("="*70 + "\n")


def main():
    """Main entry point for demos."""
    demo = DemoScenarios()
    demo.run_all_demos()


if __name__ == "__main__":
    main()
