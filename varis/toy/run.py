"""Train or evaluate VARIS on the two contextual triple-well variants."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from . import geometry, coupled, solver, decision
from ..optimization import OuterConfig

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate','decision'])
    p.add_argument('--variant',choices=['frozen','coupled'],default='frozen')
    p.add_argument('--config',type=Path,default=Path('configs/toy.json'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int)
    p.add_argument('--quick',action='store_true',help='Evaluate one context and two preferences.')
    a=p.parse_args();cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if a.variant=='coupled':
        geometry.branch_centers=coupled.branch_centers
        solver.branch_centers=coupled.branch_centers
        solver.ContextualTripleWellProblem.exact_oracle=coupled.exact_oracle
    if a.action=='train':
        device=torch.device(a.device);geometry.seed_everything(a.seed)
        model=geometry.MLP(3,6).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        steps=a.steps or cfg['training_steps'];history=[]
        for step in range(steps):
            c,x,_,centers=geometry.sample_problem(cfg['batch_size'],device,'train')
            prediction=model(torch.cat((c,x),1)).reshape(-1,3,2)
            loss=geometry.permutation_set_loss(prediction,centers)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step==0 or (step+1)%500==0 or step==steps-1:
                row={'step':step+1,'set_loss':float(loss.detach())};history.append(row);print(row,flush=True)
        torch.save({'state_dict':model.cpu().state_dict(),'variant':a.variant,'seed':a.seed},a.output/'generator.pt')
        (a.output/'training.json').write_text(json.dumps(history,indent=2));return
    checkpoint=a.checkpoint or Path(f'checkpoints/toy/{a.variant}_seed{a.seed}.pt')
    model=geometry.MLP(3,6).double();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('variant',a.variant)!=a.variant:raise ValueError('Checkpoint variant mismatch')
    model.load_state_dict(state['state_dict']);model.eval().requires_grad_(False)
    contexts=list(cfg['CONTEXTS'].items());preferences=list(cfg['PREFERENCES'].items())
    if a.quick:contexts=contexts[:1];preferences=preferences[:2]
    rows=[]
    for cname,c in contexts:
        for pname,pref in preferences:
            problem=solver.ContextualTripleWellProblem(c,a.seed);r=np.asarray(pref)
            if a.action=='decision':
                x,y=decision.optimize_query('learned_set',problem,r,{'multi':model},a.steps or cfg['decision_steps'],cfg['decision_lr'])
                metrics=decision.evaluate(problem,r,x,y)
            else:
                outer=OuterConfig(outer_steps=a.steps or cfg['outer_steps'],outer_lr=cfg['outer_lr'])
                x,y,_=solver.run_amortized(problem,r,{'multi':model},'ours_full',outer)
                metrics=solver.evaluate_result(problem,r,x,y)
            rows.append(dict(seed=a.seed,variant=a.variant,context=cname,preference=pname,split=problem.split,**metrics))
    with (a.output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary={k:float(np.mean([row[k] for row in rows])) for k in ['valid','preferred_valid','lower_gap','projected_regret']}
    summary.update(queries=len(rows),seed=a.seed,variant=a.variant)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':main()
