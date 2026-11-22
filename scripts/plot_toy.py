"""Plot the three exact responses and learned candidates at a selected query."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from varis.toy.geometry import MLP, branch_centers

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=Path("checkpoints/toy/frozen_seed0.pt"))
    p.add_argument("--output",type=Path,default=Path("outputs/toy_responses.pdf"))
    a=p.parse_args()
    model=MLP(3,6)
    model.load_state_dict(torch.load(a.checkpoint,map_location="cpu",weights_only=True)["state_dict"])
    c=torch.tensor([[.45,-.60]]); x=torch.tensor([[0.]])
    with torch.no_grad():
        centers=branch_centers(c,x)[0].numpy()
        inferred=model(torch.cat([c,x],1)).reshape(3,2).numpy()
    low=centers.min(0)-.8;high=centers.max(0)+.8
    xx,yy=np.meshgrid(np.linspace(low[0],high[0],180),np.linspace(low[1],high[1],180))
    points=np.stack([xx,yy],-1)
    gap=((points[:,:,None]-centers)**2).sum(-1).min(-1)
    fig,ax=plt.subplots(figsize=(4.6,3.6),layout="constrained")
    ax.contour(xx,yy,gap,levels=[.18**2,.1,.3,.6,1.],colors="#9ca3af",linewidths=.7)
    ax.scatter(*centers.T,marker="o",s=80,facecolors="none",edgecolors="#24654b",label="Exact lower responses")
    ax.scatter(*inferred.T,marker="x",s=45,color="#72528f",label="Generated candidates")
    ax.set(xlabel="$y_1$",ylabel="$y_2$",aspect="equal")
    ax.legend(frameon=False,fontsize=8)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output)

if __name__ == "__main__":
    main()
