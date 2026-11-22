"""Small mathematical and training-path checks, independent of external assets."""
import unittest
import torch
from varis.toy.solver import ContextualTripleWellProblem
from varis.toy.geometry import permutation_set_loss
from varis.pde.model import make_dataset
from varis.pde.physics import lower_loss, reflect
from varis.pde.inference import preference_repair_plan
from varis.qubiq.model import CompactUNet, matched_loss

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_exact_toy_responses_and_set_permutation(self):
        problem=ContextualTripleWellProblem((.45,-.6),0)
        x=problem.initial_x(0)
        centers=problem.centers(x)
        for y in centers:
            self.assertAlmostEqual(float(problem.lower_loss(x,y)),0.)
        self.assertAlmostEqual(float(permutation_set_loss(centers[None],centers[[2,0,1]][None])),0.)

    def test_reflected_pde_responses_are_valid(self):
        y,nu,obs=make_dataset(3,53000,torch.device('cpu'),'interpolation')
        self.assertLess(float(lower_loss(y,nu,obs).max()),1e-10)
        self.assertLess(float(lower_loss(reflect(y),nu,obs).max()),1e-10)

    def test_preference_cannot_admit_invalid_candidate(self):
        errors=torch.tensor([[1e-5,1e-3,.1]])
        scores=torch.tensor([[[.2,.3,-100.],[.3,.1,-100.]]])
        valid,repair,trigger,has=preference_repair_plan(errors,scores,1e-4,1e-2)
        self.assertEqual(valid.tolist(),[[0,0]])
        self.assertEqual(repair.tolist(),[[1,1]])
        self.assertEqual(trigger.tolist(),[[False,True]])
        self.assertTrue(bool(has.all()))

    def test_segmentation_set_loss_and_training_gradient(self):
        model=CompactUNet(1,6)
        image=torch.rand(2,1,32,32)
        masks=(torch.rand(2,5,32,32)>.5).float()
        logits=model(image)
        loss=matched_loss(logits,masks)
        shuffled=matched_loss(logits[:,[2,0,5,1,3,4]],masks[:,[4,2,1,0,3]])
        self.assertTrue(torch.allclose(loss,shuffled,atol=1e-6))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':
    unittest.main()
