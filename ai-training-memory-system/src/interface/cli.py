"""
Command-line interface for the AI Training Memory System.
"""
import sys
import json
from pathlib import Path
from typing import Optional
from ..core.memory import AITrainingMemory


class CLI:
    """Command-line interface for memory system."""
    
    def __init__(self):
        """Initialize CLI with memory system."""
        self.memory = AITrainingMemory()
        self.running = True
        self.commands = {
            'solve': self.cmd_solve,
            'train': self.cmd_train,
            'memory': self.cmd_memory,
            'stats': self.cmd_stats,
            'clear': self.cmd_clear,
            'export': self.cmd_export,
            'config': self.cmd_config,
            'save': self.cmd_save,
            'load': self.cmd_load,
            'help': self.cmd_help,
            'quit': self.cmd_quit,
            'exit': self.cmd_quit
        }
    
    def cmd_solve(self, args: list):
        """Solve a mathematical problem."""
        if not args:
            print("❌ Usage: solve <problem>")
            return
        
        problem = ' '.join(args)
        print(f"\n🔍 Solving: {problem}")
        
        result = self.memory.solve_problem(problem)
        
        print(f"\n✨ Answer: {result['answer']}")
        print(f"📋 Explanation: {result['explanation']}")
        print(f"⚙️  Method: {result['method']}")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"⏱️  Time: {result['time']:.4f}s")
        
        if result.get('numbers_extracted'):
            print(f"🔢 Numbers found: {result['numbers_extracted']}")
        if result.get('operation'):
            print(f"➕ Operation: {result['operation']}")
    
    def cmd_train(self, args: list):
        """Enter training mode."""
        print("\n🎓 Entering training mode...")
        self.memory.training_phase()
        print("✅ Training complete!")
        print(f"📊 Total training cycles: {self.memory.training_cycles}")
    
    def cmd_memory(self, args: list):
        """Show memory contents."""
        self.memory.show_memory()
    
    def cmd_stats(self, args: list):
        """Display performance statistics."""
        report = self.memory.performance_tracker.generate_report()
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)
        for section, data in report.items():
            print(f"\n{section.upper()}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        print("\n" + "="*60)
    
    def cmd_clear(self, args: list):
        """Clear current session data."""
        confirm = input("\n⚠️  Clear all memory? (yes/no): ").strip().lower()
        if confirm == 'yes':
            self.memory = AITrainingMemory()
            print("✅ Memory cleared!")
        else:
            print("❌ Cancelled")
    
    def cmd_export(self, args: list):
        """Export data to file."""
        if not args:
            print("❌ Usage: export <format> [filename]")
            print("   Formats: json, txt")
            return
        
        format_type = args[0].lower()
        filename = args[1] if len(args) > 1 else f"export.{format_type}"
        
        if format_type not in ['json', 'txt']:
            print("❌ Invalid format. Use: json, txt")
            return
        
        try:
            self.memory.performance_tracker.export_metrics(filename, format_type)
            print(f"✅ Exported to {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def cmd_config(self, args: list):
        """Change system configuration."""
        if not args:
            print("\n⚙️  Current Configuration:")
            print(f"  Similarity threshold: {self.memory.similarity_threshold}")
            print(f"  Duplicate check: {self.memory.duplicate_check}")
            print(f"  Max memory entries: {self.memory.max_memory_entries}")
            print(f"  Neural network: {self.memory.enable_neural_network}")
            return
        
        if len(args) < 2:
            print("❌ Usage: config <setting> <value>")
            return
        
        setting = args[0].lower()
        value = args[1]
        
        if setting == 'threshold':
            try:
                self.memory.similarity_threshold = float(value)
                print(f"✅ Similarity threshold set to {value}")
            except ValueError:
                print("❌ Invalid value. Must be a number between 0 and 1")
        
        elif setting == 'duplicate_check':
            self.memory.duplicate_check = value.lower() in ['true', '1', 'yes']
            print(f"✅ Duplicate check set to {self.memory.duplicate_check}")
        
        else:
            print(f"❌ Unknown setting: {setting}")
    
    def cmd_save(self, args: list):
        """Save memory to file."""
        filename = args[0] if args else "memory_save.pkl"
        
        try:
            # Create directory if it doesn't exist
            save_path = Path("data/saved_memories") / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.memory.save(str(save_path))
            print(f"✅ Memory saved to {save_path}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def cmd_load(self, args: list):
        """Load memory from file."""
        if not args:
            print("❌ Usage: load <filename>")
            return
        
        filename = args[0]
        load_path = Path("data/saved_memories") / filename
        
        if not load_path.exists():
            print(f"❌ File not found: {load_path}")
            return
        
        try:
            self.memory.load(str(load_path))
            print(f"✅ Memory loaded from {load_path}")
            print(f"📊 Loaded {len(self.memory.memory_documents)} memories")
        except Exception as e:
            print(f"❌ Load failed: {e}")
    
    def cmd_help(self, args: list):
        """Show help message."""
        print("\n" + "="*60)
        print("AI TRAINING MEMORY SYSTEM - COMMAND HELP")
        print("="*60)
        print("\nAvailable Commands:")
        print("  solve <problem>      - Solve a mathematical problem")
        print("  train                - Enter training mode")
        print("  memory               - Show memory contents")
        print("  stats                - Display performance statistics")
        print("  clear                - Clear current session data")
        print("  export <format>      - Export data (json/txt)")
        print("  config [setting]     - View or change configuration")
        print("  save [filename]      - Save memory to file")
        print("  load <filename>      - Load memory from file")
        print("  help                 - Show this help message")
        print("  quit/exit            - Exit the system")
        print("\nExamples:")
        print("  solve What is 5 plus 3?")
        print("  config threshold 0.8")
        print("  export json metrics.json")
        print("  save my_memory.pkl")
        print("\n" + "="*60 + "\n")
    
    def cmd_quit(self, args: list):
        """Exit the system."""
        print("\n👋 Goodbye!")
        self.running = False
    
    def run(self):
        """Run the CLI interface."""
        print("\n" + "="*60)
        print("AI TRAINING MEMORY SYSTEM")
        print("="*60)
        print("\nType 'help' for available commands")
        print("Type 'quit' or 'exit' to exit\n")
        print("="*60 + "\n")
        
        while self.running:
            try:
                user_input = input("💭 > ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:]
                
                if command in self.commands:
                    self.commands[command](args)
                else:
                    # Try to solve as a problem
                    print(f"\n🔍 Interpreting as problem to solve...")
                    self.cmd_solve(parts)
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Main entry point for CLI."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
