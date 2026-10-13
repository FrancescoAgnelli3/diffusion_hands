import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/splineeqnet'))
sys.path.insert(0,str(ROOT/'tools'))
from audit_cit_coverage_solver import energy_score, initial_residual, solve, summarize_candidates, trajectory_errors
from models.causal_innovation_transport import CausalInnovationConfig,CausalInnovationForecaster


class CoverageSolverAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_nested_best_of_k_mean_and_energy_score(self):
        target=torch.zeros(2,2,1,3)
        pred=torch.zeros(2,3,2,1,3)
        pred[:,0]+=3;pred[:,1]+=2;pred[:,2]+=1
        ade,fde=trajectory_errors(pred,target)
        self.assertTrue(torch.equal(ade,fde))
        report=summarize_candidates(pred,target,[1,2,3])
        self.assertGreater(report['best_ADE@1'],report['best_ADE@2'])
        self.assertGreater(report['best_ADE@2'],report['best_ADE@3'])
        self.assertAlmostEqual(report['candidate_mean_ADE'],float(ade.mean()))
        self.assertTrue(torch.isfinite(energy_score(pred,target)).all())
        with self.assertRaises(ValueError):summarize_candidates(pred,target,[4])

    def model(self):
        cfg=CausalInnovationConfig(input_length=4,pred_length=3,num_nodes=3,hidden_dim=16,
            num_layers=1,k_low=3,denoiser_dim=16,denoiser_depth=1,denoiser_heads=4,
            ddim_steps=4,cond_use_history=True,freeze_coarse=True,propagation='linear')
        model=CausalInnovationForecaster(cfg,{'wrist_index':1})
        d=model.diffusion;d.calibrated.fill_(True);d.persistence.fill_(.7)
        d.radius.fill_(.5);d.innovation_scale.fill_(.2)
        with torch.no_grad():d.denoiser.out.bias.fill_(.03)
        return model.eval()

    def test_cartesian_euler_matches_native_sampler(self):
        torch.manual_seed(9);model=self.model();history=torch.randn(2,4,3,3)
        with torch.no_grad():
            coarse=model.coarse(history)
            initial=initial_residual(model,history,coarse,17)
            audited=solve(model,history,coarse,initial,'euler',4)
            native=model.predict(history,coarse_future=coarse,seed=17)
        torch.testing.assert_close(audited,native,atol=2e-6,rtol=1e-5)
        self.assertTrue((audited[:,:,1]==0).all())

    def test_heun_is_deterministic_and_does_not_mutate_model(self):
        model=self.model();history=torch.randn(1,4,3,3)
        state=copy.deepcopy(model.state_dict())
        with torch.no_grad():
            coarse=model.coarse(history);initial=initial_residual(model,history,coarse,3)
            one=solve(model,history,coarse,initial,'heun',3)
            two=solve(model,history,coarse,initial,'heun',3)
        self.assertTrue(torch.equal(one,two))
        self.assertTrue(all(isinstance(v,torch.nn.parameter.UninitializedParameter)
                            or torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
        with self.assertRaises(ValueError):solve(model,history,coarse,initial,'bad',3)


if __name__=='__main__':unittest.main()
