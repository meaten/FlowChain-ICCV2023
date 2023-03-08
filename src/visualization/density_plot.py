from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

 
def plot_density(x: np.array, y: np.array, p: np.array, path: Path, traj = None) -> None:
    plt.pcolormesh(x, y, p.reshape(x.shape),
                   shading='auto',
                   cmap=plt.cm.get_cmap("Greens"),
                   norm=matplotlib.colors.Normalize())
    if traj is not None:
        obs, gt = traj
        sns.lineplot(x=gt[:, 0], y=gt[:, 1],
                     color='black', marker='o')
        sns.lineplot(x=obs[:, 0], y=obs[:, 1],
                     color='green', marker='o')
        
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    from scipy.stats import gaussian_kde

    offsets = [[-2.5, -0.5], [0.0, 1.0]]
    nbins=300
    xi, yi = np.mgrid[-10:10:nbins*1j, -10:10:nbins*1j]
    zi_list = []
    for offset in offsets:
        # create data
        x = np.random.normal(size=500, scale=0.5) + offset[0]
        y = np.random.normal(size=500, scale=0.5) + offset[1]
        
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        k = gaussian_kde([x,y])
        zi_list.append(k(np.vstack([xi.flatten(), yi.flatten()])))

    z = np.zeros_like(zi_list[0])
    for i, zi in enumerate(zi_list):
        z += zi
        plot_density(xi, yi, z/(i+1), path=f"{i}.png")
    

