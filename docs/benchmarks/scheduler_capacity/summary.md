# Fixed-search-limit capacity study

Limits apply to each CP-SAT search. Construction, application and validation times are reference measurements and do not determine success. FEASIBLE is not an optimality proof. UNKNOWN is not an infeasibility proof. Delays use virtual minutes; computation uses real seconds.

| Pattern | Size | Phase | Limit s | Valid / runs | Search median / p95 / max s | Status |
| --- | ---: | --- | ---: | ---: | --- | --- |
| contention | 2 | io | 1 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| contention | 2 | scan | 1 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| contention | 2 | scan | 60 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| contention | 4 | scan | 1 | 1/1 | 1.000 / 1.000 / 1.000 | FEASIBLE |
| contention | 4 | scan | 60 | 1/1 | 1.005 / 1.005 / 1.005 | OPTIMAL |
| contention | 8 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| contention | 8 | scan | 60 | 1/1 | 60.004 / 60.004 / 60.004 | FEASIBLE |
| contention | 16 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| contention | 16 | scan | 60 | 1/1 | 60.003 / 60.003 / 60.003 | FEASIBLE |
| contention | 32 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| contention | 32 | scan | 60 | 1/1 | 60.004 / 60.004 / 60.004 | FEASIBLE |
| contention | 64 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| contention | 64 | scan | 60 | 1/1 | 60.004 / 60.004 / 60.004 | FEASIBLE |
| contention | 128 | scan | 1 | 1/1 | 1.002 / 1.002 / 1.002 | FEASIBLE |
| contention | 128 | scan | 60 | 1/1 | 60.002 / 60.002 / 60.002 | FEASIBLE |
| contention | 256 | operation | 1 | 1/1 | 18.014 / 18.014 / 18.014 | FEASIBLE |
| contention | 256 | operation | 60 | 1/1 | 1079.966 / 1079.966 / 1079.966 | FEASIBLE |
| contention | 256 | scan | 1 | 5/5 | 1.002 / 1.003 / 1.003 | FEASIBLE |
| contention | 256 | scan | 60 | 5/5 | 60.002 / 60.003 / 60.003 | FEASIBLE |
| contention | 512 | scan | 1 | 0/5 | 1.003 / 1.003 / 1.004 | UNKNOWN |
| contention | 512 | scan | 60 | 5/5 | 60.005 / 60.007 / 60.007 | FEASIBLE |
| heterogeneous | 2 | io | 1 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| heterogeneous | 2 | scan | 1 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| heterogeneous | 2 | scan | 60 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| heterogeneous | 4 | scan | 1 | 1/1 | 0.008 / 0.008 / 0.008 | OPTIMAL |
| heterogeneous | 4 | scan | 60 | 1/1 | 0.008 / 0.008 / 0.008 | OPTIMAL |
| heterogeneous | 8 | scan | 1 | 1/1 | 1.000 / 1.000 / 1.000 | FEASIBLE |
| heterogeneous | 8 | scan | 60 | 1/1 | 3.459 / 3.459 / 3.459 | OPTIMAL |
| heterogeneous | 16 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| heterogeneous | 16 | scan | 60 | 1/1 | 60.001 / 60.001 / 60.001 | FEASIBLE |
| heterogeneous | 32 | scan | 1 | 1/1 | 0.991 / 0.991 / 0.991 | FEASIBLE |
| heterogeneous | 32 | scan | 60 | 1/1 | 60.002 / 60.002 / 60.002 | FEASIBLE |
| heterogeneous | 64 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| heterogeneous | 64 | scan | 60 | 1/1 | 60.002 / 60.002 / 60.002 | FEASIBLE |
| heterogeneous | 128 | scan | 1 | 1/1 | 1.000 / 1.000 / 1.000 | FEASIBLE |
| heterogeneous | 128 | scan | 60 | 1/1 | 60.003 / 60.003 / 60.003 | FEASIBLE |
| heterogeneous | 256 | scan | 1 | 5/5 | 1.002 / 1.003 / 1.003 | FEASIBLE |
| heterogeneous | 256 | scan | 60 | 5/5 | 60.005 / 60.205 / 60.255 | FEASIBLE |
| heterogeneous | 512 | operation | 1 | 1/1 | 18.047 / 18.047 / 18.047 | FEASIBLE |
| heterogeneous | 512 | operation | 60 | 1/1 | 1079.620 / 1079.620 / 1079.620 | FEASIBLE |
| heterogeneous | 512 | scan | 1 | 5/5 | 1.004 / 1.004 / 1.004 | FEASIBLE |
| heterogeneous | 512 | scan | 60 | 5/5 | 60.005 / 60.009 / 60.010 | FEASIBLE |
| many_candidates | 2 | io | 1 | 1/1 | 0.005 / 0.005 / 0.005 | OPTIMAL |
| many_candidates | 2 | scan | 1 | 1/1 | 0.005 / 0.005 / 0.005 | OPTIMAL |
| many_candidates | 2 | scan | 60 | 1/1 | 0.005 / 0.005 / 0.005 | OPTIMAL |
| many_candidates | 4 | scan | 1 | 1/1 | 0.012 / 0.012 / 0.012 | OPTIMAL |
| many_candidates | 4 | scan | 60 | 1/1 | 0.012 / 0.012 / 0.012 | OPTIMAL |
| many_candidates | 8 | scan | 1 | 1/1 | 0.044 / 0.044 / 0.044 | OPTIMAL |
| many_candidates | 8 | scan | 60 | 1/1 | 0.044 / 0.044 / 0.044 | OPTIMAL |
| many_candidates | 16 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| many_candidates | 16 | scan | 60 | 1/1 | 60.001 / 60.001 / 60.001 | FEASIBLE |
| many_candidates | 32 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| many_candidates | 32 | scan | 60 | 1/1 | 60.001 / 60.001 / 60.001 | FEASIBLE |
| many_candidates | 64 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| many_candidates | 64 | scan | 60 | 1/1 | 60.002 / 60.002 / 60.002 | FEASIBLE |
| many_candidates | 128 | operation | 1 | 0/1 | 3.702 / 3.702 / 3.702 | FEASIBLE, UNKNOWN |
| many_candidates | 128 | operation | 60 | 1/1 | 1080.039 / 1080.039 / 1080.039 | FEASIBLE |
| many_candidates | 128 | scan | 1 | 5/5 | 1.002 / 1.002 / 1.002 | FEASIBLE |
| many_candidates | 128 | scan | 60 | 1/1 | 59.998 / 59.998 / 59.998 | FEASIBLE |
| many_candidates | 256 | scan | 1 | 0/5 | 0.832 / 0.852 / 0.855 | UNKNOWN |
| many_candidates | 256 | scan | 60 | 5/5 | 60.004 / 60.005 / 60.005 | FEASIBLE |
| many_candidates | 512 | scan | 1 | 0/5 | 1.012 / 1.013 / 1.013 | UNKNOWN |
| many_candidates | 512 | scan | 60 | 5/5 | 60.008 / 60.009 / 60.009 | FEASIBLE |
| mixed_penalties | 2 | io | 1 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| mixed_penalties | 2 | scan | 1 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| mixed_penalties | 2 | scan | 60 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| mixed_penalties | 4 | scan | 1 | 1/1 | 0.895 / 0.895 / 0.895 | OPTIMAL |
| mixed_penalties | 4 | scan | 60 | 1/1 | 0.893 / 0.893 / 0.893 | OPTIMAL |
| mixed_penalties | 8 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| mixed_penalties | 8 | scan | 60 | 1/1 | 60.001 / 60.001 / 60.001 | FEASIBLE |
| mixed_penalties | 16 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| mixed_penalties | 16 | scan | 60 | 1/1 | 60.003 / 60.003 / 60.003 | FEASIBLE |
| mixed_penalties | 32 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| mixed_penalties | 32 | scan | 60 | 1/1 | 60.003 / 60.003 / 60.003 | FEASIBLE |
| mixed_penalties | 64 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| mixed_penalties | 64 | scan | 60 | 1/1 | 60.004 / 60.004 / 60.004 | FEASIBLE |
| mixed_penalties | 128 | operation | 1 | 0/1 | 1.575 / 1.575 / 1.575 | FEASIBLE, UNKNOWN |
| mixed_penalties | 128 | operation | 60 | 1/1 | 1080.035 / 1080.035 / 1080.035 | FEASIBLE |
| mixed_penalties | 128 | scan | 1 | 5/5 | 1.005 / 1.005 / 1.005 | FEASIBLE |
| mixed_penalties | 128 | scan | 60 | 1/1 | 60.003 / 60.003 / 60.003 | FEASIBLE |
| mixed_penalties | 256 | scan | 1 | 0/5 | 1.006 / 1.010 / 1.010 | UNKNOWN |
| mixed_penalties | 256 | scan | 60 | 5/5 | 60.004 / 60.008 / 60.009 | FEASIBLE |
| mixed_penalties | 512 | scan | 1 | 0/5 | 1.022 / 1.024 / 1.024 | UNKNOWN |
| mixed_penalties | 512 | scan | 60 | 5/5 | 60.009 / 60.010 / 60.010 | FEASIBLE |
| replanning | 2 | io | 1 | 1/1 | 0.002 / 0.002 / 0.002 | OPTIMAL |
| replanning | 2 | scan | 1 | 1/1 | 0.003 / 0.003 / 0.003 | OPTIMAL |
| replanning | 2 | scan | 60 | 1/1 | 0.002 / 0.002 / 0.002 | OPTIMAL |
| replanning | 4 | scan | 1 | 1/1 | 0.014 / 0.014 / 0.014 | OPTIMAL |
| replanning | 4 | scan | 60 | 1/1 | 0.014 / 0.014 / 0.014 | OPTIMAL |
| replanning | 8 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| replanning | 8 | scan | 60 | 1/1 | 60.003 / 60.003 / 60.003 | FEASIBLE |
| replanning | 16 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| replanning | 16 | scan | 60 | 1/1 | 60.003 / 60.003 / 60.003 | FEASIBLE |
| replanning | 32 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| replanning | 32 | scan | 60 | 1/1 | 60.002 / 60.002 / 60.002 | FEASIBLE |
| replanning | 64 | scan | 1 | 1/1 | 1.001 / 1.001 / 1.001 | FEASIBLE |
| replanning | 64 | scan | 60 | 1/1 | 60.004 / 60.004 / 60.004 | FEASIBLE |
| replanning | 128 | scan | 1 | 1/1 | 1.002 / 1.002 / 1.002 | FEASIBLE |
| replanning | 128 | scan | 60 | 1/1 | 60.003 / 60.003 / 60.003 | FEASIBLE |
| replanning | 256 | operation | 1 | 1/1 | 18.008 / 18.008 / 18.008 | FEASIBLE |
| replanning | 256 | operation | 60 | 1/1 | 1079.999 / 1079.999 / 1079.999 | FEASIBLE |
| replanning | 256 | scan | 1 | 5/5 | 1.002 / 1.002 / 1.002 | FEASIBLE |
| replanning | 256 | scan | 60 | 5/5 | 60.001 / 60.002 / 60.002 | FEASIBLE |
| replanning | 512 | scan | 1 | 0/5 | 1.002 / 1.002 / 1.003 | UNKNOWN |
| replanning | 512 | scan | 60 | 5/5 | 60.005 / 60.006 / 60.006 | FEASIBLE |

p95 uses linear interpolation; five repetitions do not establish an operational guarantee.
Memory includes the Python/OR-Tools imports. Utilization describes the pending schedule, not a calibrated facility throughput model.
