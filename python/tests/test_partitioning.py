"""Tests for partitioning module"""

import unittest


class TestPartitioning(unittest.TestCase):
    """Test partitioning functionality"""

    def test_uniform_partitioning(self):
        """Test uniform partition creation"""
        # signal = [0.0, 1.0, 2.0, 3.0, 4.0]
        # partitioner = UniformPartitioner(num_symbols=2)
        # partition = partitioner.compute_partition(signal)
        # self.assertEqual(partition.num_regions, 2)
        pass

    def test_maximum_entropy_partitioning(self):
        """Test maximum entropy partitioning"""
        # signal = [1.0, 2.0, 1.0, 3.0, 2.0]
        # partitioner = MaximumEntropyPartitioner(num_symbols=3)
        # partition = partitioner.compute_partition(signal)
        # self.assertGreater(len(partition.boundaries), 0)
        pass


if __name__ == '__main__':
    unittest.main()
