# Physical-position implementation lock (before new inference)
Preserve original full static split window order andbatchsize256, including3.7m/s
windows in forward computation so batching matches accepted inference exactly;
scientific outputfocus remains6.1/8.5. No autocast. Recompute eachacceptedflight's
five normalizedmetric means andperhorizonmeans, require rtol5e-4/atol1e-6 for
FP32hardware numericalagreement. Also recordmaximumperwindowabsolute differences;
perwindow bitwiseequality is not assumed. Do not replace original metrics or
relax thresholds after seeing failures. Position estimates usefloat32 cumulative
predictedvelocity as originalintegrator, translatedbyrecordedposition; compare
physicalrecordedposition andEuler-integratedloggedvelocity separately.

Full-window GPU pilots: rosterindices0and6 (dataset0seed70 originalFrozen and
newfixedauxiliarycontext). Separatepilotnamespace; bothmustreproduceaccepted
error summaries beforeformal60modelaudit. Pilot outcomes do notchange tolerance,
modelrosterorhorizons. Positionarrays retained for all57000primarywindwindows
permodel, alongwithflight/windowidentities andperflightmetrics for auditability.
