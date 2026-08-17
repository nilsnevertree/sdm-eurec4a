# %%
### ------------------------- FUNCTION DEFINITIONS ------------------------- ###
def parse_arguments():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path2sdmeurec4aCLEO", type=Path, help="Absolute path to sdmeurec4aCLEO directory"
    )
    parser.add_argument(
        "path2clusters", type=Path, help="Absolute path to .zarr Cleo dataset"
    )
    return parser.parse_args()

def check_paths(path2sdmeurec4aCLEO, path2clusters, script2run):
    assert (
            path2sdmeurec4aCLEO
        ).is_dir(), "sdmeurec4aCleo directory not found"

    assert (
            path2clusters
        ).is_dir(), "path to clusters' directory doesn't exist"

    assert (
            script2run
        ).is_file(), "script creating DSD datasets for a single cluster doesn't exit"

    print("path2sdmeurec4aCLEO:", path2sdmeurec4aCLEO)
    print("path2clusters:", path2clusters)
    print("script2run", script2run)

def find_clusters(path2clusters):
    import glob

    clusters = glob.glob(str(path2clusters / "cluster_*"))

    return clusters
# %% ### ------------------------- MAIN PROGRAM ------------------------- ### 
def _run_cluster(args):
    from pathlib import Path
    import subprocess
    
    script2run, path2sdmeurec4aCLEO, cpath = args
    cpath = Path(cpath)
    path2zarr = cpath / "eurec4a1d_sol.zarr"
    path2setup = cpath / "config" / "eurec4a1d_setup.txt"
    path2grid = cpath / "share" / "eurec4a1d_ddimlessGBxboundaries.dat"
    path2output = cpath / "processed"

    subprocess.run(["python", str(script2run), str(path2sdmeurec4aCLEO), str(path2zarr), str(path2setup), str(path2grid), str(path2output)], check=True)

def main(path2sdmeurec4aCLEO, path2clusters):
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor

    script2run = Path(__file__).resolve().parent / "create_dsd_datasets.py"

    check_paths(path2sdmeurec4aCLEO, path2clusters, script2run)

    cluster_paths = find_clusters(path2clusters)
    print("clusters_found:", [Path(cpath).name for cpath in sorted(cluster_paths)])

    with ThreadPoolExecutor() as executor:
        executor.map(_run_cluster, [(script2run, path2sdmeurec4aCLEO, cpath) for cpath in cluster_paths])

# %% Run main
if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.path2sdmeurec4aCLEO,
        args. path2clusters
    )

# # %% EXAMPLE FOR TESTING
# from pathlib import Path

# path2sdmeurec4aCLEO = Path("/Users/yoctoyotta1024/Documents/c1_springsummer2024/nils_masters/rain-evap-nils/sdm-eurec4a-CLEO/")
# path2clusters = Path("/Users/yoctoyotta1024/Downloads/rain-evap-nils/sdm-eurec4a-CLEO/data/output_v4.2/condensation/")

# main(
#     path2sdmeurec4aCLEO,
#     path2clusters
#     )
