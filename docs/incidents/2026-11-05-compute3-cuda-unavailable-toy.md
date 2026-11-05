# 2026-11-05 — TOY validation ran without a usable CUDA device

Compute3's NVIDIA driver was unavailable during the TOY physics runs. Startup logs in
`~/dojo-data/toy-held/g3-on-notrace.log` contain `cudaErrorNoDevice`, `No CUDA devices found`, and failed
`PxCudaContextManager` creation. A read-only check in the existing RoboDojo Python returned
`torch.cuda.is_available() == False`. CONV independently recorded the outage in its
[work journal](../journal/2026-11-05-conv-waist-intercept.md).

The runner continued with software physics and the completed rows did not carry an `infra_failure` field.
I initially described these progressing episodes as validation; that interpretation was withdrawn when the startup
logs were inspected. The newly generated scores and dense replays are diagnostic smoke only; they do not establish
GPU/G3 performance or a camera gain. The report separates them from historical saved pre-outage replays.

The treatment simulator and queued baseline were stopped by their own PIDs (737984, 746153, 746144); subsequent
`ps -p` found none. All earlier TOY replay processes had exited or were stopped by their own recorded PIDs.
No other process, driver, package, system setting or machine was changed. Compute4 stayed read-only.

Follow-up: machine owner restores the driver under the runbook; rerun paired validation on fresh output dirs after
checking startup logs. Harden the experiment infra classification so failed CUDA context creation cannot pass
unmarked as a healthy run. This PR adds the landmine to BACKLOG; it does not change shared runtime hardware policy.
