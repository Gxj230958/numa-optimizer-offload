import copy
import unittest

import torch

from offload import PipelinedAdamW


class OffloadTest(unittest.TestCase):
    def test_accumulation_clipping_and_resume(self):
        torch.manual_seed(17)
        base = torch.nn.Sequential(torch.nn.Linear(13, 31), torch.nn.GELU(), torch.nn.Linear(31, 7))
        inputs = [(torch.randn(5, 13), torch.randn(5, 7)) for _ in range(12)]
        reference = copy.deepcopy(base)
        expected = torch.optim.AdamW(reference.parameters(), lr=0.003, weight_decay=0.1)
        actual = copy.deepcopy(base)
        optimizer = PipelinedAdamW(actual.parameters(), bucket_mb=0.001, depth=3,
                                  lr=0.003, weight_decay=0.1, max_grad_norm=0.07)
        resumed = None
        for step in range(4):
            expected.zero_grad()
            optimizer.zero_grad()
            if resumed:
                resumed[1].zero_grad()
            for x, y in inputs[step*3:(step+1)*3]:
                ((reference(x)-y).square().mean()/3).backward()
                ((actual(x)-y).square().mean()/3).backward()
                if resumed:
                    ((resumed[0](x)-y).square().mean()/3).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.07)
            expected.step()
            optimizer.step()
            if resumed:
                resumed[1].step()
            for a, b in zip(reference.parameters(), actual.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-7)
            if resumed:
                for a, b in zip(actual.parameters(), resumed[0].parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            if step == 1:
                model = copy.deepcopy(base)
                opt = PipelinedAdamW(model.parameters(), bucket_mb=0.001, depth=1)
                opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                resumed = model, opt
        optimizer.close()
        resumed[1].close()

    def test_nonfinite_does_not_update(self):
        p = torch.nn.Parameter(torch.ones(10))
        optimizer = PipelinedAdamW([p])
        p.grad = torch.full_like(p, float("inf"))
        with self.assertRaises(FloatingPointError):
            optimizer.step()
        torch.testing.assert_close(p, torch.ones(10))
        self.assertFalse(optimizer.buckets[0].optimizer.state)
        optimizer.close()


if __name__ == "__main__":
    unittest.main()
