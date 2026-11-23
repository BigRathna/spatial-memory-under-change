# Implicit memory: candidate failure mechanisms

`REPORT.md` states that the 32-unit MLP has no usable operating point in the
object-count sweep. The mechanisms below are consistent with the code and the
sweep, but none is isolated by an experiment in this repository, so they are
hypotheses rather than results. They were moved here from the report to keep
the report to two pages.

1. **Time resolution against the update budget.** One step changes the `t`
   input by `1/64`. Separating adjacent steps would need weights of magnitude
   around 64, which 50 SGD steps at learning rate 0.01 do not build. A query
   can therefore return a ghost of an earlier position. Test: freeze `t` or add
   a per-step bias, and compare on the same budget and seeds.
2. **Capacity.** Thirty localized bumps may need about two tanh half-spaces
   each, more than a single 32-unit layer supplies. Test: a capacity-matched
   variant (more units, a second layer, or an id embedding) at 3 and 30 objects
   with the same 50-step budget and at least five seeds.
3. **Scan resolution floor.** The query scans 2,048 integer cells. A 2-cell
   success radius covers 33 cells, about 1.6% of the grid by chance. The
   reported 0.027 current success is 2 hits in 75 queries, above that floor but
   not statistically distinguishable from chance. Test: more queries or more
   sweep seeds.

A fourth candidate is not a mechanism but a missing measurement: the sampler
uses recency weighting (`exp(-age / 2)`), and `HANDOFF.md` used to report a lift
at 5 objects. No result file backs that comparison, so it is not evidence here.
Test: an A/B run with recency weighting on and off, committed as a CSV.
