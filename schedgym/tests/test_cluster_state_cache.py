import unittest

from .. import cluster as clstr
from .. import job, pool


class TestClusterStateCache(unittest.TestCase):
    def setUp(self):
        self.cluster = clstr.Cluster(4, 8)

    def make_job(self, job_id=1, processors=2, memory=3):
        j = job.Job(
            job_id=job_id, requested_processors=processors, requested_memory=memory
        )
        j.resources.processors = pool.IntervalTree(
            [pool.Interval(0, processors, job_id)]
        )
        j.resources.memory = pool.IntervalTree([pool.Interval(0, memory, job_id)])
        return j

    def test_state_is_cached_between_reads_when_unchanged(self):
        first = self.cluster.state
        second = self.cluster.state

        self.assertIs(first, second)

    def test_allocate_invalidates_cached_state(self):
        job_ = self.make_job()

        first = self.cluster.state
        second = self.cluster.state
        self.assertIs(first, second)

        self.cluster.allocate(job_)
        third = self.cluster.state

        self.assertIsNot(second, third)

    def test_free_invalidates_cached_state(self):
        job_ = self.make_job()
        self.cluster.allocate(job_)

        first = self.cluster.state
        second = self.cluster.state
        self.assertIs(first, second)

        self.cluster.free(job_)
        third = self.cluster.state

        self.assertIsNot(second, third)

    def test_clone_starts_with_independent_cache(self):
        clone = self.cluster.clone()

        original_first = self.cluster.state
        original_second = self.cluster.state
        clone_first = clone.state
        clone_second = clone.state

        self.assertIs(original_first, original_second)
        self.assertIs(clone_first, clone_second)
        self.assertIsNot(original_first, clone_first)

    def test_state_content_matches_resources(self):
        job_ = self.make_job(job_id=7, processors=2, memory=3)
        self.cluster.allocate(job_)

        expected = (
            (2, 2, {(0, 2): 7}),
            (5, 3, {(0, 3): 7}),
        )

        self.assertEqual(expected, self.cluster.state)

    def test_ignore_memory_state_is_single_tuple_and_cached(self):
        cluster = clstr.Cluster(4, 8, ignore_memory=True)

        first = cluster.state
        second = cluster.state

        self.assertIs(first, second)
        self.assertEqual(((4, 0, {}),), first)
