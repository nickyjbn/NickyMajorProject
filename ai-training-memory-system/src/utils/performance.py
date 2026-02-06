"""
Performance tracking and metrics.
"""
import time
import json
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict


class PerformanceTracker:
    """
    Track and analyze performance metrics for the memory system.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        self.total_queries = 0
        self.memory_hits = 0
        self.similarity_matches = 0
        self.failed_queries = 0
        
        self.solve_times: List[float] = []
        self.retrieval_times: List[float] = []
        self.computation_times: List[float] = []
        
        self.query_history: List[Dict[str, Any]] = []
        self.query_methods = defaultdict(int)
        
        self.start_time = datetime.now()
    
    def record_query(
        self,
        method: str,
        solve_time: float,
        success: bool,
        retrieval_time: float = 0.0,
        computation_time: float = 0.0
    ):
        """
        Record a query execution.
        
        Args:
            method: Method used ('memory_hit', 'similarity', 'computation', 'failed')
            solve_time: Total time to solve
            success: Whether query was successful
            retrieval_time: Time spent retrieving from memory
            computation_time: Time spent computing
        """
        self.total_queries += 1
        self.solve_times.append(solve_time)
        
        if method == 'memory_hit':
            self.memory_hits += 1
        elif method == 'similarity':
            self.similarity_matches += 1
        elif not success:
            self.failed_queries += 1
        
        if retrieval_time > 0:
            self.retrieval_times.append(retrieval_time)
        if computation_time > 0:
            self.computation_times.append(computation_time)
        
        self.query_methods[method] += 1
        
        self.query_history.append({
            'timestamp': datetime.now(),
            'method': method,
            'solve_time': solve_time,
            'success': success,
            'retrieval_time': retrieval_time,
            'computation_time': computation_time
        })
    
    def calculate_hit_rate(self) -> float:
        """
        Calculate memory hit rate.
        
        Returns:
            Hit rate as percentage
        """
        if self.total_queries == 0:
            return 0.0
        return (self.memory_hits / self.total_queries) * 100
    
    def calculate_average_time(self, times: List[float]) -> float:
        """Calculate average time from list."""
        if not times:
            return 0.0
        return sum(times) / len(times)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with performance metrics
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            'summary': {
                'total_queries': self.total_queries,
                'memory_hits': self.memory_hits,
                'similarity_matches': self.similarity_matches,
                'failed_queries': self.failed_queries,
                'hit_rate': f"{self.calculate_hit_rate():.2f}%",
                'uptime_seconds': uptime
            },
            'timing': {
                'avg_solve_time': f"{self.calculate_average_time(self.solve_times):.4f}s",
                'avg_retrieval_time': f"{self.calculate_average_time(self.retrieval_times):.4f}s",
                'avg_computation_time': f"{self.calculate_average_time(self.computation_times):.4f}s",
                'total_solve_time': f"{sum(self.solve_times):.4f}s"
            },
            'methods': dict(self.query_methods),
            'efficiency': {
                'queries_per_second': self.total_queries / uptime if uptime > 0 else 0
            }
        }
        
        return report
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """
        Export metrics to file.
        
        Args:
            filepath: Output file path
            format: Export format ('json', 'csv', 'txt')
        """
        report = self.generate_report()
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'txt':
            with open(filepath, 'w') as f:
                f.write("=== Performance Report ===\n\n")
                for section, data in report.items():
                    f.write(f"{section.upper()}:\n")
                    for key, value in data.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reset(self):
        """Reset all metrics."""
        self.__init__()
