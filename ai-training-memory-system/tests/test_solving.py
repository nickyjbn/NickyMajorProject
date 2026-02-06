"""
Unit tests for solver components.
"""
import pytest
import numpy as np
from src.solvers.rule_based import (
    extract_numbers, identify_operation, solve_addition,
    solve_subtraction, solve_multiplication, solve_division,
    solve_with_rules
)
from src.solvers.neural_network import MathSolverNN, NeuralNetworkSolver
from src.solvers.hybrid import HybridSolver


class TestNumberExtraction:
    """Test number extraction from text."""
    
    def test_extract_simple_numbers(self):
        """Test extracting simple integers."""
        numbers = extract_numbers("What is 5 plus 3?")
        assert numbers == [5.0, 3.0]
    
    def test_extract_decimal_numbers(self):
        """Test extracting decimal numbers."""
        numbers = extract_numbers("Calculate 3.14 times 2.5")
        assert 3.14 in numbers
        assert 2.5 in numbers
    
    def test_extract_negative_numbers(self):
        """Test extracting negative numbers."""
        numbers = extract_numbers("What is -5 plus 10?")
        assert -5.0 in numbers
        assert 10.0 in numbers
    
    def test_no_numbers(self):
        """Test text with no numbers."""
        numbers = extract_numbers("Hello world")
        assert len(numbers) == 0


class TestOperationIdentification:
    """Test operation identification."""
    
    def test_addition_keywords(self):
        """Test identifying addition."""
        assert identify_operation("What is 5 plus 3?") == 'addition'
        assert identify_operation("Add 10 and 20") == 'addition'
        assert identify_operation("Calculate the sum of 5 and 7") == 'addition'
    
    def test_subtraction_keywords(self):
        """Test identifying subtraction."""
        assert identify_operation("What is 10 minus 3?") == 'subtraction'
        assert identify_operation("Subtract 5 from 20") == 'subtraction'
    
    def test_multiplication_keywords(self):
        """Test identifying multiplication."""
        assert identify_operation("What is 5 times 3?") == 'multiplication'
        assert identify_operation("Multiply 7 by 6") == 'multiplication'
    
    def test_division_keywords(self):
        """Test identifying division."""
        assert identify_operation("What is 20 divided by 4?") == 'division'
        assert identify_operation("Divide 100 by 5") == 'division'
    
    def test_unknown_operation(self):
        """Test unknown operation."""
        assert identify_operation("What is the answer?") == 'unknown'


class TestMathematicalOperations:
    """Test mathematical operation functions."""
    
    def test_addition_problems(self):
        """Test addition solver."""
        assert solve_addition([5, 3]) == 8
        assert solve_addition([10, 20, 30]) == 60
        assert solve_addition([0, 0]) == 0
    
    def test_subtraction_problems(self):
        """Test subtraction solver."""
        assert solve_subtraction([10, 3]) == 7
        assert solve_subtraction([100, 25, 10]) == 65
        assert solve_subtraction([5]) == 5
    
    def test_multiplication_problems(self):
        """Test multiplication solver."""
        assert solve_multiplication([5, 3]) == 15
        assert solve_multiplication([2, 3, 4]) == 24
        assert solve_multiplication([10, 0]) == 0
    
    def test_division_problems(self):
        """Test division solver."""
        assert solve_division([20, 4]) == 5.0
        assert solve_division([100, 5]) == 20.0
        assert solve_division([10, 2]) == 5.0
    
    def test_division_by_zero(self):
        """Test division by zero handling."""
        result = solve_division([10, 0])
        assert result is None


class TestRuleBasedSolver:
    """Test complete rule-based solving."""
    
    def test_solve_addition(self):
        """Test solving addition problems."""
        result = solve_with_rules("What is 5 plus 3?")
        
        assert result['success'] == True
        assert result['answer'] == 8
        assert result['operation'] == 'addition'
        assert len(result['numbers_extracted']) == 2
    
    def test_solve_subtraction(self):
        """Test solving subtraction problems."""
        result = solve_with_rules("Calculate 10 minus 4")
        
        assert result['success'] == True
        assert result['answer'] == 6
        assert result['operation'] == 'subtraction'
    
    def test_solve_multiplication(self):
        """Test solving multiplication problems."""
        result = solve_with_rules("What is 7 times 6?")
        
        assert result['success'] == True
        assert result['answer'] == 42
        assert result['operation'] == 'multiplication'
    
    def test_solve_division(self):
        """Test solving division problems."""
        result = solve_with_rules("Divide 20 by 4")
        
        assert result['success'] == True
        assert result['answer'] == 5.0
        assert result['operation'] == 'division'
    
    def test_mixed_operations(self):
        """Test problems with multiple numbers."""
        result = solve_with_rules("Add 10, 20, and 30")
        
        assert result['success'] == True
        assert result['answer'] == 60
        assert len(result['numbers_extracted']) == 3


class TestNeuralNetwork:
    """Test neural network components."""
    
    def test_model_architecture(self):
        """Test neural network architecture."""
        model = MathSolverNN(input_dim=384)
        
        # Check layers exist
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')
        assert hasattr(model, 'fc4')
        
        # Check input dimension
        assert model.input_dim == 384
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        model = MathSolverNN(input_dim=384)
        
        # Create dummy input
        x = np.random.randn(1, 384).astype(np.float32)
        import torch
        x_tensor = torch.FloatTensor(x)
        
        # Forward pass
        output = model(x_tensor)
        
        assert output.shape == (1, 1)
    
    def test_solver_initialization(self):
        """Test neural network solver initialization."""
        solver = NeuralNetworkSolver(input_dim=384, learning_rate=0.001)
        
        assert solver.model is not None
        assert solver.optimizer is not None
        assert solver.criterion is not None
        assert solver.trained == False
    
    def test_training(self):
        """Test neural network training."""
        solver = NeuralNetworkSolver()
        
        # Create dummy training data
        X_train = np.random.randn(20, 384)
        y_train = np.random.randn(20)
        
        # Train
        history = solver.train(X_train, y_train, epochs=2, batch_size=4)
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == 2
        assert solver.trained == True
    
    def test_prediction(self):
        """Test making predictions."""
        solver = NeuralNetworkSolver()
        
        # Train on dummy data
        X_train = np.random.randn(20, 384)
        y_train = np.random.randn(20)
        solver.train(X_train, y_train, epochs=2)
        
        # Predict
        X_test = np.random.randn(5, 384)
        predictions = solver.predict(X_test)
        
        assert predictions.shape == (5,)


class TestHybridSolver:
    """Test hybrid solver."""
    
    def test_hybrid_without_nn(self):
        """Test hybrid solver without neural network."""
        hybrid = HybridSolver(nn_solver=None)
        
        result = hybrid.solve("What is 5 plus 3?")
        
        assert 'final_answer' in result
        assert result['rule_based_success'] == True
    
    def test_hybrid_with_nn(self):
        """Test hybrid solver with neural network."""
        nn_solver = NeuralNetworkSolver()
        
        # Train on dummy data
        X_train = np.random.randn(20, 384)
        y_train = np.random.randn(20)
        nn_solver.train(X_train, y_train, epochs=2)
        
        hybrid = HybridSolver(nn_solver=nn_solver)
        embedding = np.random.randn(384)
        
        result = hybrid.solve("What is 5 plus 3?", embedding=embedding)
        
        assert 'final_answer' in result
        assert 'nn_prediction' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
