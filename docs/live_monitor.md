# Live progress monitor

Open **http://127.0.0.1:8765** on the Mac running the experiment. The page refreshes every three seconds and displays actual checkpoint counts, completed source examples, remaining responses, valid/failed responses, observed speed, GPU memory, and estimated completion times in Tehran time. The estimate adapts to observed speed; the later bound uses a 25% margin over the slowest of the recent, overall and smoke averages.

The monitor is read-only and serves only its page and progress summary. It binds to the loopback interface, so it is not published online. Closing the browser page does not stop the experiment. A disconnected monitor is not proof that model inference stopped. The completed model-response count is distinct from completion of statistical analysis, qualitative review and the PDF.

If the monitor process needs restarting:

```sh
.venv/bin/python scripts/monitor_server.py --port 8765
```

## Execution and resume

The monitor and inference run as separate processes. Completed responses are checkpointed individually. An interrupted run resumes matching completed jobs; missing responses remain missing observations. For a new local run, scoped `caffeinate` can prevent idle sleep while inference runs. Closing the lid or powering off can still interrupt the GPU job.

## Recorded study artifacts

The completed study's three response-checkpoint files, run manifests, raw CSVs, analysis tables, plots and report are tracked in this repository. Model weights, installed environments and temporary hardware-test responses are excluded. The monitor is an optional local tool; the published results can be read and reproduced without it.
