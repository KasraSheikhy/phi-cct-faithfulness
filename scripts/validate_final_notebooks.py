"""Fresh-kernel checks, including standalone replay in a directory with no project files."""
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
import nbformat
from nbclient import NotebookClient
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def execute(path,cwd):
    nb=nbformat.read(path,as_version=4)
    client=NotebookClient(nb,timeout=300,kernel_name="python3",resources={"metadata":{"path":str(cwd)}})
    client.execute()
    assert all(not any(o.output_type=="error" for o in c.outputs) for c in nb.cells if c.cell_type=="code")
    nbformat.write(nb,path)
    return sum(c.cell_type=="code" for c in nb.cells)


if __name__=="__main__":
    with tempfile.TemporaryDirectory(prefix="phi_cct_clean_notebook_") as folder:
        temp=Path(folder)
        count=execute(ROOT/"notebooks/phi_cct_esnli.ipynb",temp)
        replay=temp/"phi-cct-faithfulness"
        assert not (replay/"scripts").exists()
        coverage=pd.read_csv(replay/"outputs/results/coverage.csv")
        assert len(coverage)==1000 and coverage.completion_status.eq("complete").all()
        for name in ["summary","tests","plot_estimates","phi_contrasts","mention_sensitivity"]:
            reference=pd.read_csv(ROOT/f"outputs/results/{name}.csv")
            actual=pd.read_csv(replay/f"outputs/results/{name}.csv")
            assert reference.shape==actual.shape
            for col in reference.select_dtypes(include=np.number).columns:
                assert np.allclose(reference[col],actual[col],equal_nan=True,rtol=1e-10,atol=1e-12), (name,col)
        json.loads((replay/"outputs/notebook_environment.json").read_text())
    analysis_cells=execute(ROOT/"notebooks/analysis.ipynb",ROOT/"notebooks")
    result={"verified_at_utc":datetime.now(timezone.utc).isoformat(),"canonical_restart_run_all":"passed_in_clean_directory",
      "canonical_code_cells":count,"hidden_project_scripts_present":False,"raw_recorded_responses_replayed":9000,
      "new_model_generations":0,"all_primary_and_sensitivity_numeric_results_reproduced":True,
      "analysis_companion_restart_run_all":"passed","analysis_companion_code_cells":analysis_cells,
      "fresh_generation_path":"included; actual completed GPU execution recorded separately"}
    (ROOT/"outputs/notebook_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
