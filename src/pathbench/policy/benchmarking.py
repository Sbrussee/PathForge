"""
This code should implement benchmarking based on base classes.
1) Define a BenchmarkingPolicy class that inherits from PolicyBase.
2) Implement abstract methods for setting up, running, and reporting benchmarks.
3) Based on search space in config, find all possible configurations to benchmark.
4) Execute benchmarks for each configuration and collect results.

Results should be saved to csv formats which can then be analyzed later.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathbench.policy.base import PolicyBase

class BenchmarkingPolicy(PolicyBase):
    """Policy for benchmarking different configurations."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.results: List[Dict[str, Any]] = []
    
    @abstractmethod
    def setup_benchmark(self) -> None:
        """Set up the benchmarking environment."""
        pass
    
    @abstractmethod
    def run_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a benchmark for a given configuration and return the results."""
        pass
    
    @abstractmethod
    def report_results(self) -> None:
        """Report the collected benchmark results."""
        pass
    
    def execute(self) -> None:
        """Execute the benchmarking process."""
        self.setup_benchmark()
        
        search_space = self.config.get("search_space", [])
        for configuration in search_space:
            result = self.run_benchmark(configuration)
            self.results.append(result)
        
        self.report_results()