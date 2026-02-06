"""
Hybrid solver combining rule-based and neural network approaches.
"""
from typing import Dict, Any, Optional
import numpy as np
from .rule_based import solve_with_rules, extract_numbers, identify_operation
from .neural_network import NeuralNetworkSolver


class HybridSolver:
    """
    Combines rule-based and neural network solving approaches.
    Uses rule-based solving as primary method and NN for enhancement.
    """
    
    def __init__(self, nn_solver: Optional[NeuralNetworkSolver] = None):
        """
        Initialize hybrid solver.
        
        Args:
            nn_solver: Neural network solver instance (optional)
        """
        self.nn_solver = nn_solver
        self.use_nn = nn_solver is not None and nn_solver.trained
    
    def solve(self, problem: str, embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Solve problem using hybrid approach.
        
        Args:
            problem: Problem text
            embedding: Problem embedding vector (optional, for NN)
            
        Returns:
            Dictionary with solution details
        """
        # First try rule-based solving
        rule_result = solve_with_rules(problem)
        
        result = {
            'problem': problem,
            'rule_based_answer': rule_result.get('answer'),
            'rule_based_success': rule_result.get('success', False),
            'numbers_extracted': rule_result.get('numbers_extracted', []),
            'operation': rule_result.get('operation', 'unknown'),
            'explanation': rule_result.get('explanation', ''),
            'method': 'rule_based',
            'confidence': 1.0 if rule_result.get('success') else 0.5
        }
        
        # Try neural network if available and embedding provided
        if self.use_nn and embedding is not None:
            try:
                nn_prediction = self.nn_solver.predict(embedding.reshape(1, -1))[0]
                result['nn_prediction'] = float(nn_prediction)
                
                # If rule-based succeeded, use it; otherwise use NN
                if result['rule_based_success']:
                    result['final_answer'] = result['rule_based_answer']
                    result['method'] = 'hybrid_rule_primary'
                else:
                    result['final_answer'] = nn_prediction
                    result['method'] = 'hybrid_nn_fallback'
                    result['confidence'] = 0.6
                    result['explanation'] = f"Neural network prediction: {nn_prediction:.2f}"
            except Exception as e:
                result['nn_error'] = str(e)
                result['final_answer'] = result['rule_based_answer']
        else:
            result['final_answer'] = result['rule_based_answer']
        
        return result
