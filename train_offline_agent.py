import argparse
import d3rlpy
from d3rlpy.datasets import MDPDataset
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/offline_dataset.h5")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--model-path", type=str, default="models/offline_agent.pt")
    args = parser.parse_args()

    # Load the offline dataset
    with h5py.File(args.dataset, 'r') as f:
        observations = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]

    dataset = MDPDataset(observations, actions, rewards, terminals)

    # Create the CQL agent
    cql = d3rlpy.algos.CQLConfig().create()

    # Train the agent
    cql.fit(
        dataset,
        n_epochs=args.n_epochs,
    )

    # Save the trained model
    cql.save_model(args.model_path)

    print(f"Offline agent saved to {args.model_path}")

if __name__ == "__main__":
    main()
