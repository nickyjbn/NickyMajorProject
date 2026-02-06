"""
Visualization tools for the AI Training Memory System.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from datetime import datetime


class Visualizer:
    """Create charts and visualizations for system performance."""
    
    def __init__(self, memory_system=None):
        """
        Initialize visualizer.
        
        Args:
            memory_system: AITrainingMemory instance
        """
        self.memory_system = memory_system
    
    def plot_memory_growth(self, save_path: str = None):
        """
        Plot memory growth over time.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.memory_system or not self.memory_system.memory_documents:
            print("No memory data available")
            return
        
        timestamps = [doc.metadata.get('timestamp', datetime.now()) 
                     for doc in self.memory_system.memory_documents]
        memory_counts = list(range(1, len(timestamps) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, memory_counts, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Number of Memories')
        plt.title('Memory Growth Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_solve_time_distribution(self, save_path: str = None):
        """
        Plot distribution of solve times.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.memory_system or not self.memory_system.solve_times:
            print("No solve time data available")
            return
        
        solve_times = self.memory_system.solve_times
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(solve_times, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Solve Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Solve Times')
        ax1.grid(True, alpha=0.3)
        
        # Time series
        ax2.plot(solve_times, marker='o', linestyle='-', linewidth=1, markersize=4)
        ax2.set_xlabel('Query Number')
        ax2.set_ylabel('Solve Time (seconds)')
        ax2.set_title('Solve Times Over Queries')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_performance_metrics(self, save_path: str = None):
        """
        Plot performance metrics bar chart.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.memory_system:
            print("No memory system available")
            return
        
        report = self.memory_system.performance_tracker.generate_report()
        
        methods = list(report['methods'].keys())
        counts = list(report['methods'].values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, counts, edgecolor='black', alpha=0.7)
        
        # Color bars differently
        colors = ['green', 'blue', 'orange', 'red']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Method')
        plt.ylabel('Count')
        plt.title('Query Methods Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_similarity_heatmap(
        self,
        queries: List[str],
        save_path: str = None
    ):
        """
        Plot similarity heatmap between queries.
        
        Args:
            queries: List of query strings
            save_path: Optional path to save figure
        """
        if not self.memory_system:
            print("No memory system available")
            return
        
        if len(queries) < 2:
            print("Need at least 2 queries for heatmap")
            return
        
        # Generate embeddings
        embeddings = [self.memory_system.embedder.embed(q) for q in queries]
        
        # Calculate similarity matrix
        n = len(queries)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Cosine similarity
                    dot_product = np.dot(embeddings[i], embeddings[j])
                    norm_i = np.linalg.norm(embeddings[i])
                    norm_j = np.linalg.norm(embeddings[j])
                    similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Similarity Score')
        
        # Add labels
        short_queries = [q[:30] + '...' if len(q) > 30 else q for q in queries]
        plt.xticks(range(n), short_queries, rotation=45, ha='right')
        plt.yticks(range(n), short_queries)
        
        # Add values in cells
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='black', fontsize=8)
        
        plt.title('Query Similarity Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_learning_curve(
        self,
        training_history: Dict[str, List[float]],
        save_path: str = None
    ):
        """
        Plot neural network training learning curve.
        
        Args:
            training_history: Dictionary with 'train_loss' and optionally 'val_loss'
            save_path: Optional path to save figure
        """
        if not training_history or 'train_loss' not in training_history:
            print("No training history available")
            return
        
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        plt.plot(epochs, training_history['train_loss'], 
                marker='o', linestyle='-', linewidth=2, label='Training Loss')
        
        if 'val_loss' in training_history and training_history['val_loss']:
            plt.plot(epochs, training_history['val_loss'],
                    marker='s', linestyle='--', linewidth=2, label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Neural Network Training Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def create_dashboard(self, save_path: str = None):
        """
        Create comprehensive dashboard with multiple plots.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.memory_system:
            print("No memory system available")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Memory growth
        if self.memory_system.memory_documents:
            ax1 = fig.add_subplot(gs[0, :])
            timestamps = [doc.metadata.get('timestamp', datetime.now()) 
                         for doc in self.memory_system.memory_documents]
            memory_counts = list(range(1, len(timestamps) + 1))
            ax1.plot(timestamps, memory_counts, marker='o', linestyle='-', linewidth=2)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Number of Memories')
            ax1.set_title('Memory Growth Over Time')
            ax1.grid(True, alpha=0.3)
        
        # Solve time distribution
        if self.memory_system.solve_times:
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.hist(self.memory_system.solve_times, bins=20, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Solve Time (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Solve Time Distribution')
            ax2.grid(True, alpha=0.3)
        
        # Performance metrics
        report = self.memory_system.performance_tracker.generate_report()
        if report['methods']:
            ax3 = fig.add_subplot(gs[1, 1])
            methods = list(report['methods'].keys())
            counts = list(report['methods'].values())
            ax3.bar(methods, counts, edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Method')
            ax3.set_ylabel('Count')
            ax3.set_title('Query Methods')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Statistics text
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        stats_text = f"""
        SYSTEM STATISTICS
        
        Total Queries: {self.memory_system.total_queries}
        Memory Hits: {self.memory_system.memory_hit_count}
        Hit Rate: {(self.memory_system.memory_hit_count / self.memory_system.total_queries * 100) if self.memory_system.total_queries > 0 else 0:.2f}%
        
        Total Memories: {len(self.memory_system.memory_documents)}
        Unique Questions: {len(self.memory_system.question_history)}
        Training Cycles: {self.memory_system.training_cycles}
        
        Average Solve Time: {sum(self.memory_system.solve_times) / len(self.memory_system.solve_times) if self.memory_system.solve_times else 0:.4f}s
        """
        
        ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('AI Training Memory System Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def main():
    """Demo visualization capabilities."""
    from ..core.memory import AITrainingMemory
    
    # Create sample memory system
    memory = AITrainingMemory()
    
    # Add some sample data
    problems = [
        "What is 5 plus 3?",
        "Calculate 10 minus 4",
        "What is 7 times 6?",
        "Divide 20 by 4"
    ]
    
    for problem in problems:
        memory.solve_problem(problem)
    
    # Create visualizer and plot
    viz = Visualizer(memory)
    viz.create_dashboard()


if __name__ == "__main__":
    main()
