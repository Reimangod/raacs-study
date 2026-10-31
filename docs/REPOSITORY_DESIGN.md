# Sources for the repository design

Inspected 2026-10-31. These sources inform packaging, not proof of RAACS performance.

| Source | What is adopted |
|---|---|
| https://github.com/jrm874/sqd_data_repository | Separate integrals, classical references, raw samples and CI results; name the implementation version |
| https://zenodo.org/records/15324153 | Archive a fixed data release separately from a moving GitHub branch |
| https://github.com/kevinsung/lucj | Separate source, generation, analysis, plots; provide dependency lock |
| https://github.com/Qiskit/qiskit-addon-sqd | Testable library with explicit documentation and release version |
| https://github.com/mafaldaramoa/ceo-adapt-vqe | Preserve actual upstream source and attribution; never substitute current main for the executed version |
| https://researcher-resources.acs.org/publish/author_guidelines?coden=jctcce | Data Availability Statement and accessible research evidence; not a requirement to publish every private file |
| https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github | Keep large raw data out of ordinary Git history |
| https://zenodo.org/help/versioning | Cite a specific dataset version, not just the latest moving project |

The adopted design intentionally has no new multi-stage scientific authorization
framework, no automatic molecular reruns and no artificial performance claim.
The minimum deliverables are code/configuration identity, faithful numerical
records, traceable raw files, clear limitations and affordable reproduction.
