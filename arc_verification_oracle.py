#!/usr/bin/env python3
"""
Secure Verification Oracle for ARC AGI
Verifies solutions without exposing answers
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

class VerificationOracle:
    """Secure oracle for verifying puzzle solutions"""
    
    def __init__(self):
        self.attempts = {}  # Track attempts per puzzle
        self.verified_solutions = {}  # Store verified correct solutions
        self.max_attempts_per_puzzle = 10  # Prevent brute forcing
        
    def verify_solution(self, puzzle_id: str, submitted_output: List[List[int]], 
                       expected_output: List[List[int]] = None) -> Dict:
        """
        Verify a submitted solution against the expected output
        
        Args:
            puzzle_id: The puzzle identifier
            submitted_output: The solution to verify
            expected_output: The correct answer (if available)
            
        Returns:
            Verification result without exposing the actual answer
        """
        # Track attempts
        if puzzle_id not in self.attempts:
            self.attempts[puzzle_id] = []
        
        # Check attempt limit
        if len(self.attempts[puzzle_id]) >= self.max_attempts_per_puzzle:
            return {
                "verified": False,
                "error": "Maximum verification attempts exceeded",
                "attempts_remaining": 0,
                "hint": None
            }
        
        # Record attempt
        attempt_record = {
            "timestamp": datetime.now().isoformat(),
            "submission_hash": self._hash_grid(submitted_output)
        }
        self.attempts[puzzle_id].append(attempt_record)
        
        # If no expected output provided, we can't verify
        if expected_output is None:
            return {
                "verified": None,
                "message": "No verification data available for this puzzle",
                "attempts_remaining": self.max_attempts_per_puzzle - len(self.attempts[puzzle_id])
            }
        
        # Verify dimensions
        if (len(submitted_output) != len(expected_output) or 
            (submitted_output and expected_output and 
             len(submitted_output[0]) != len(expected_output[0]))):
            
            size_hint = self._get_size_hint(submitted_output, expected_output)
            return {
                "verified": False,
                "correct": False,
                "feedback": "Incorrect grid dimensions",
                "hint": size_hint,
                "attempts_remaining": self.max_attempts_per_puzzle - len(self.attempts[puzzle_id]),
                "accuracy": 0.0
            }
        
        # Calculate accuracy without revealing the answer
        accuracy = self._calculate_accuracy(submitted_output, expected_output)
        is_correct = accuracy == 1.0
        
        if is_correct:
            # Store verified solution
            self.verified_solutions[puzzle_id] = {
                "verified_at": datetime.now().isoformat(),
                "attempts": len(self.attempts[puzzle_id]),
                "solution_hash": self._hash_grid(submitted_output)
            }
            
            return {
                "verified": True,
                "correct": True,
                "feedback": "Perfect! Solution is correct!",
                "attempts_used": len(self.attempts[puzzle_id]),
                "accuracy": 1.0
            }
        else:
            # Provide helpful feedback without revealing answer
            feedback = self._generate_feedback(submitted_output, expected_output, accuracy)
            
            return {
                "verified": True,
                "correct": False,
                "feedback": feedback["message"],
                "hint": feedback["hint"],
                "accuracy": accuracy,
                "attempts_remaining": self.max_attempts_per_puzzle - len(self.attempts[puzzle_id])
            }
    
    def _hash_grid(self, grid: List[List[int]]) -> str:
        """Create a hash of the grid for tracking"""
        grid_str = json.dumps(grid, sort_keys=True)
        return hashlib.sha256(grid_str.encode()).hexdigest()[:16]
    
    def _calculate_accuracy(self, submitted: List[List[int]], expected: List[List[int]]) -> float:
        """Calculate accuracy without revealing the answer"""
        if not submitted or not expected:
            return 0.0
        
        total_cells = len(expected) * len(expected[0])
        if total_cells == 0:
            return 0.0
        
        correct_cells = 0
        for i in range(len(expected)):
            for j in range(len(expected[0])):
                if submitted[i][j] == expected[i][j]:
                    correct_cells += 1
        
        return correct_cells / total_cells
    
    def _get_size_hint(self, submitted: List[List[int]], expected: List[List[int]]) -> str:
        """Provide size hint without revealing exact dimensions"""
        sub_h, sub_w = len(submitted), len(submitted[0]) if submitted else 0
        exp_h, exp_w = len(expected), len(expected[0]) if expected else 0
        
        if sub_h < exp_h or sub_w < exp_w:
            return "Output grid is too small"
        elif sub_h > exp_h or sub_w > exp_w:
            return "Output grid is too large"
        else:
            return "Check grid dimensions"
    
    def _generate_feedback(self, submitted: List[List[int]], expected: List[List[int]], 
                          accuracy: float) -> Dict:
        """Generate helpful feedback without revealing the answer"""
        feedback = {
            "message": "",
            "hint": None
        }
        
        if accuracy >= 0.9:
            feedback["message"] = "Very close! Check the details carefully."
            feedback["hint"] = "Almost there - review the pattern one more time"
        elif accuracy >= 0.7:
            feedback["message"] = "Good progress! Some cells need correction."
            feedback["hint"] = "You have the right idea, but some transformations are incomplete"
        elif accuracy >= 0.5:
            feedback["message"] = "Partially correct. Review the transformation pattern."
            feedback["hint"] = "About half the cells are correct - check your pattern logic"
        elif accuracy >= 0.3:
            feedback["message"] = "Some elements are correct. Reconsider the pattern."
            feedback["hint"] = "Some progress, but the main pattern might need rethinking"
        else:
            feedback["message"] = "Not quite right. Try analyzing the training examples again."
            feedback["hint"] = "Review the training examples for the core pattern"
        
        # Add specific hints based on common errors (without revealing answer)
        color_distribution = self._analyze_color_distribution(submitted, expected)
        if color_distribution["very_different"]:
            feedback["hint"] += ". Check color mappings."
        
        pattern_analysis = self._analyze_patterns(submitted, expected)
        if pattern_analysis["symmetry_issue"]:
            feedback["hint"] += ". Consider symmetry."
        
        return feedback
    
    def _analyze_color_distribution(self, submitted: List[List[int]], expected: List[List[int]]) -> Dict:
        """Analyze color distribution differences"""
        sub_colors = set(cell for row in submitted for cell in row)
        exp_colors = set(cell for row in expected for cell in row)
        
        # Don't reveal exact colors, just if they're very different
        return {
            "very_different": len(sub_colors.symmetric_difference(exp_colors)) > len(exp_colors) / 2
        }
    
    def _analyze_patterns(self, submitted: List[List[int]], expected: List[List[int]]) -> Dict:
        """Analyze pattern differences without revealing specifics"""
        # Check if symmetry properties match (without revealing which symmetry)
        sub_h_sym = self._has_horizontal_symmetry(submitted)
        exp_h_sym = self._has_horizontal_symmetry(expected)
        
        sub_v_sym = self._has_vertical_symmetry(submitted)
        exp_v_sym = self._has_vertical_symmetry(expected)
        
        return {
            "symmetry_issue": (sub_h_sym != exp_h_sym) or (sub_v_sym != exp_v_sym)
        }
    
    def _has_horizontal_symmetry(self, grid: List[List[int]]) -> bool:
        """Check horizontal symmetry"""
        if not grid:
            return False
        for i in range(len(grid) // 2):
            if grid[i] != grid[-(i+1)]:
                return False
        return True
    
    def _has_vertical_symmetry(self, grid: List[List[int]]) -> bool:
        """Check vertical symmetry"""
        if not grid:
            return False
        for row in grid:
            for j in range(len(row) // 2):
                if row[j] != row[-(j+1)]:
                    return False
        return True
    
    def get_verification_stats(self, puzzle_id: str) -> Dict:
        """Get verification statistics for a puzzle"""
        if puzzle_id not in self.attempts:
            return {
                "puzzle_id": puzzle_id,
                "attempts": 0,
                "solved": False,
                "attempts_remaining": self.max_attempts_per_puzzle
            }
        
        return {
            "puzzle_id": puzzle_id,
            "attempts": len(self.attempts[puzzle_id]),
            "solved": puzzle_id in self.verified_solutions,
            "attempts_remaining": max(0, self.max_attempts_per_puzzle - len(self.attempts[puzzle_id])),
            "solved_at": self.verified_solutions.get(puzzle_id, {}).get("verified_at")
        }
    
    def reset_attempts(self, puzzle_id: str):
        """Reset attempts for a puzzle (admin function)"""
        if puzzle_id in self.attempts:
            del self.attempts[puzzle_id]
        if puzzle_id in self.verified_solutions:
            del self.verified_solutions[puzzle_id]
    
    def check_if_solved(self, puzzle_id: str) -> bool:
        """Check if a puzzle has been solved"""
        return puzzle_id in self.verified_solutions
    
    def get_all_stats(self) -> Dict:
        """Get overall statistics"""
        total_puzzles_attempted = len(self.attempts)
        total_puzzles_solved = len(self.verified_solutions)
        total_attempts = sum(len(attempts) for attempts in self.attempts.values())
        
        avg_attempts = (total_attempts / total_puzzles_solved) if total_puzzles_solved > 0 else 0
        
        return {
            "total_puzzles_attempted": total_puzzles_attempted,
            "total_puzzles_solved": total_puzzles_solved,
            "total_attempts": total_attempts,
            "average_attempts_per_solved": avg_attempts,
            "solve_rate": (total_puzzles_solved / total_puzzles_attempted) if total_puzzles_attempted > 0 else 0
        }

# Global oracle instance
verification_oracle = VerificationOracle()