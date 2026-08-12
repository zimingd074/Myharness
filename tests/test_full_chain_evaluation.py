import gc
import os
import tempfile
import unittest

from evoagent.evaluation_benchmark import generate_controlled_pr_cases
from evoagent.evaluation_harness import EndToEndEvaluationHarness
from evoagent.full_chain_evaluation import QueuedServiceReviewer, build_service


class FullChainEvaluationTests(unittest.TestCase):
    def test_case_runs_through_queued_review_service(self):
        case = generate_controlled_pr_cases()[0]
        handle, database_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        reviewer = QueuedServiceReviewer(build_service(database_path), "full-chain-candidate")
        try:
            result = EndToEndEvaluationHarness().run(reviewer, [case])
        finally:
            reviewer.close()
            gc.collect()
            os.unlink(database_path)
        self.assertEqual("queued-review-service", result["execution_path"])
        self.assertEqual(["SUCCESS"], reviewer.task_states)
        self.assertEqual(1, result["metrics"]["tp"])


if __name__ == "__main__":
    unittest.main()
