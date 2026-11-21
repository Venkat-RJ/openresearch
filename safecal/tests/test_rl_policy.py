"""SafeCal neural probe RL smoke tests. Skip if optional PyTorch not installed."""
import unittest

try:
    import torch
    import numpy as np
    from safecal import SafeCalEnv
    from bayes_id import BayesianID
    from train_probe_rl import ProbeNet, observation, action_logits, play, STOP, N_PROBES
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Install safecal-v1[rl] for neural policy checks")
class NeuralPolicyTests(unittest.TestCase):
    def test_observations(self):
        env = SafeCalEnv(900000)
        model = BayesianID(samples=4)
        x = observation(model, env)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(len(x), 42)

    def test_action_mask(self):
        env = SafeCalEnv(900000)
        model = BayesianID(samples=4)
        net = ProbeNet(features=len(observation(model,env)))
        logits,_ = action_logits(net,observation(model,env),
                                 set(env.available_actions()),False)
        self.assertEqual(int(torch.argmax(logits)), STOP)
        self.assertTrue((logits[:N_PROBES]<-1e8).all())

    def test_training_episode(self):
        env = SafeCalEnv(1009)
        model = BayesianID(samples=4)
        net = ProbeNet(features=len(observation(model,env)))
        torch.manual_seed(777)
        episode = play(1009,net,save_grad=True)
        self.assertGreaterEqual(episode.reward, 0.0)
        self.assertLessEqual(episode.cost,4.5)
        self.assertLessEqual(len(episode.probes),2)
        if episode.logprob is not None:
            (-episode.logprob).backward()
            self.assertTrue(any(p.grad is not None for p in net.parameters()))


if __name__=="__main__": unittest.main()
