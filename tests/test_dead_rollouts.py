# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import tempfile
import unittest
from pathlib import Path

from harness2.core import schedule
from harness2.core.ledger import Ledger
from harness2.demo import DemoAdapter, DemoProposer


class DyingDemoAdapter(DemoAdapter):
    """The demo executor, with the rollouts named in DIE crashing before they produce a run.

    Like the benchmark adapters, a missing run directory reads as no deliverable.
    """

    DIE = ("/base2", "/inv1")

    def execute(self, task_id, harness_dir, run_id, *, timeout=None):
        if run_id.endswith(self.DIE):
            raise RuntimeError(f"executor crashed for {run_id.rsplit('/', 1)[-1]}")
        return super().execute(task_id, harness_dir, run_id, timeout=timeout)

    def deliverable_text(self, run_dir):
        return super().deliverable_text(run_dir) if self.delivered(run_dir) else ""

    def deliverable_files(self, run_dir):
        return super().deliverable_files(run_dir) if self.delivered(run_dir) else {}


class DeadRolloutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _run(self, adapter, k):
        return schedule.run_domain(
            "accounting",
            adapter.tasks(),
            k,
            adapter=adapter,
            prop=DemoProposer(adapter.spec.channels["proposer"]),
            runs_root=adapter.spec.runs_root,
            library_root=self.root / "library" / str(k),
        )

    def _ledgers(self, adapter, k):
        for task_id in adapter.tasks():
            summary_path = adapter.spec.runs_root / f"demo-oc-agg{k}" / task_id / "summary.json"
            summary = json.loads(summary_path.read_text())
            for mode in ("agg", "seq"):
                yield mode, summary["modes"][mode]["ledger"]

    def test_dead_rollouts_are_named_with_their_reason(self):
        adapter = DyingDemoAdapter(self.root)
        result = self._run(adapter, 2)
        self.assertEqual((result["done"], result["failed"]), (2, 0))
        for mode, ledger in self._ledgers(adapter, 2):
            with self.subTest(mode=mode):
                self.assertEqual(ledger["R"], 8)
                self.assertEqual(set(ledger["dead_rollouts"]), {"base2", "inv1"})
                self.assertIn("executor crashed for base2", ledger["dead_rollouts"]["base2"])

    def test_live_rollouts_leave_the_record_empty(self):
        adapter = DemoAdapter(self.root)
        self._run(adapter, 1)
        for _mode, ledger in self._ledgers(adapter, 1):
            self.assertEqual(ledger["dead_rollouts"], {})

    def test_a_dead_rollout_still_counts_toward_the_budget(self):
        ledger = Ledger(mode="agg", task_id="t", k=1)
        ledger.rollout("base1", dead="RuntimeError: boom")
        for label in ("inv1", "dom1", "agg1"):
            ledger.rollout(label, 0.1)
        for label in ("inv1", "dom1", "agg1"):
            ledger.session(label)
        ledger.assert_identity()
        self.assertEqual(ledger.dead_rollouts, {"base1": "RuntimeError: boom"})
        self.assertEqual(ledger.to_dict()["dead_rollouts"], {"base1": "RuntimeError: boom"})


if __name__ == "__main__":
    unittest.main()
