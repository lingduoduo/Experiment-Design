"""Learn a causal graph and run Bayesian RCA on synthetic0.pkl."""

import argparse
import platform
import pprint
import subprocess
from pathlib import Path


def start_java():
    """Initialize the Java backend imported alongside PyRCA's PC algorithm."""
    import jpype
    import pyrca

    if jpype.isJVMStarted():
        return
    jar = (Path(pyrca.__file__).parent / "thirdparty" / "pytetrad"
           / "resources" / "tetrad-gui-current-launch.jar")
    if not jar.is_file():
        raise FileNotFoundError(f"PyRCA's Tetrad archive is missing: {jar}")
    if platform.system() == "Darwin":
        java_home = subprocess.check_output(
            ["/usr/libexec/java_home", "-a", platform.machine()], text=True
        ).strip()
        jvm_path = str(Path(java_home) / "lib" / "server" / "libjvm.dylib")
    else:
        jvm_path = jpype.getDefaultJVMPath()
    jpype.startJVM(jvm_path, "-Xmx2g", classpath=[str(jar)])


def main():
    import dill
    import pandas as pd

    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=directory / "synthetic0.pkl")
    parser.add_argument("--model-dir", type=Path, default=directory / "models")
    parser.add_argument("--anomalies", nargs="+", default=["X1"])
    args = parser.parse_args()

    # Load the same trusted simulation dataset used by the RCA notebook.
    with args.data.open("rb") as input_file:
        data = dill.load(input_file)
    samples = data["data"]["data"]
    training_samples = int(data["data"]["num_samples"])
    if not 0 < training_samples < len(samples):
        raise ValueError("The dataset must contain normal and abnormal samples.")
    columns = [f"X{i + 1}" for i in range(samples.shape[1])]
    unknown = set(args.anomalies) - set(columns)
    if unknown:
        parser.error(f"Unknown anomaly metrics: {', '.join(sorted(unknown))}")
    normal_df = pd.DataFrame(samples[:training_samples], columns=columns)

    start_java()
    from pyrca.graphs.causal.pc import PC
    from pyrca.analyzers.bayesian import BayesianNetwork, BayesianNetworkConfig

    graph = PC(PC.config_class(run_pdag2dag=True)).train(normal_df)
    model = BayesianNetwork(BayesianNetworkConfig(graph=graph))
    model.train(normal_df)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    graph.to_pickle(args.model_dir / "adjacency_df.pkl")
    model.save(str(args.model_dir))
    print(f"Trained on {len(normal_df)} normal samples; saved model to {args.model_dir}")
    pprint.pprint(model.find_root_causes(args.anomalies).to_list())


if __name__ == "__main__":
    main()
