The original data is from more than 20 years ago, so unfortunately, we could not find the very raw data, and there is some unrecoverable inconsistencies. What we can find is some xls files, likely produced by some stats software. The following are our best guess with the help of CLAUDE.

TRIAL1.xls, TRIAL2.xls, TRIAL3.xls are three trials processed separately.

rr7.xls aggregates the second and the third trial,discarding the first trial as practice, by pooling them together and run separately, not by averaging TRIAL2.xls and TRIAL3.xls.

*_contingency_tables.csv and *_counts_by_subjects.csv are implied from p(yes) and Q under the design constraints, with the help of CLAUDE, where Q is smoothed by +0.5 to each cell in the contingency table. However, the design constraints are not always met. Most subjects have the correct sum of targets, others have plus or minus one. However, for repeated lures, theoretically they should correspond to same item (si) and intact pair (int). Currently, only TRIAL2.xls satisfies with most subjects, other xls satisfies with only about a half subjects.

The repeated-lure block is duplicated between TRIAL1 and TRIAL3. Likely unsuable. Also related to the above problem.

The p1_pi column in TRIAL*.xls is corrupt. It's mutually inconsistent with the Q_pi and p2_pi in the same row. If you instead solve for Test-1 p(yes) from Q_pi and p2_pi, the implied value pools to rr7's p1_pi for 60 of 61 subjects. rr7 has the right number. This also resolves a substantive absurdity: the trial files claim 0.78 for a Test-1 pair probe in pi but 0.91 for the identical probe type in int; rr7 gives 0.91 and 0.89.

After all, rr7.xls is mostly self-consistent, with the only weakness that the repeated lures may not correspond so well with the design. We will adopt that as the published version.