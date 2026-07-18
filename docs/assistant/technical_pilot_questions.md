# NEAT Assistant Technical-Knowledge Pilot

Use these questions in the **NEAT AI Assistant** after restarting NEAT. They
test the newly curated technical guidance with wording that differs from the
automated evaluation set.

Run one session at a time. For every answer:

1. read the answer and its verified sources;
2. choose **Helpful** or **Not helpful** in the assistant panel;
3. if Not helpful, record a short note under that question describing whether
   the problem was an incorrect fact, missing detail, unclear wording,
   irrelevant source or unsafe scientific claim.

Do not paste raw experimental data, confidential metadata, credentials or
personally identifying information into a question.

## Session 1 — Preprocessing rules

- [ ] **TP-01:** One pixel is much brighter than everything around it. What
  numerical rule does Clean use before calling it a positive spike?
  - Reviewer note:

- [ ] **TP-02:** Clean found an invalid pixel, but every value in its immediate
  neighbourhood is also invalid. What will it do next?
  - Reviewer note:

- [ ] **TP-03:** Can I use Window half `n=150` and Adjacent `m=20` for
  normalisation? What ranges does NEAT allow?
  - Reviewer note:

- [ ] **TP-04:** I do not want normalisation to average neighbouring wavelength
  frames. Which Adjacent value should I use, and is that the default?
  - Reviewer note:

- [ ] **TP-05:** Are preprocessing masks and Data Post-Processing masks handled
  identically when they contain values such as 0.5?
  - Reviewer note:

## Session 2 — Wavelength, phases and model parameters

- [ ] **TP-06:** My manual calibration values are time of flight in
  milliseconds. How are delay and flight path used to obtain wavelength?
  - Reviewer note:

- [ ] **TP-07:** With several manual wavelength anchors, what happens before
  the first anchor and after the last one?
  - Reviewer note:

- [ ] **TP-08:** If I import a two-column wavelength/intensity profile, will
  NEAT normalise it again?
  - Reviewer note:

- [ ] **TP-09:** Please give me NEAT's corrected built-in crystal structure and
  reference lattice value for copper and beta titanium.
  - Reviewer note:

- [ ] **TP-10:** Which of `s`, `t` and `eta` describes sample microstructure,
  instrument broadening and neutron-pulse edge shape?
  - Reviewer note:

## Session 3 — Fitting and mapping behaviour

- [ ] **TP-11:** I fixed `eta`, and its uncertainty is shown as NaN. Does that
  mean the uncertainty is zero or the fit failed?
  - Reviewer note:

- [ ] **TP-12:** Is NEAT's plotted residual fit-minus-data or data-minus-fit?
  Does the optimiser use per-point counting errors as weights?
  - Reviewer note:

- [ ] **TP-13:** My selected phase produces seven valid edges. Will individual
  fitting and batch mapping ignore the last two?
  - Reviewer note:

- [ ] **TP-14:** Inside each mapping box, are detector pixels averaged or
  summed, and at which pixel is the result stored?
  - Reviewer note:

- [ ] **TP-15:** I pressed Stop while the final mapping box was being fitted.
  Should NEAT write a partial CSV result?
  - Reviewer note:

## Session 4 — Results and scientific boundaries

- [ ] **TP-16:** For reflection `(1,1,0)`, should an individual ungridded CSV
  contain `d_11` or `d_110`?
  - Reviewer note:

- [ ] **TP-17:** My imported result grid is 300×420. What dimensions should
  Save image write to FITS?
  - Reviewer note:

- [ ] **TP-18:** What warning and numerical effect should I expect if a
  post-processing mask contains 0.2 and 0.8 instead of only 0 and 1?
  - Reviewer note:

- [ ] **TP-19:** Can I calculate strain from `d_unc_110`, `s_110` or only from
  `d_110`? What sign does positive microstrain have?
  - Reviewer note:

- [ ] **TP-20:** My detector was binned and I need millimetre coordinates. Can
  I assume NEAT's mm switch is calibrated correctly, and can I infer stress
  directly from the strain map?
  - Reviewer note:

## Acceptance criteria

The technical pilot is ready for review when:

- all 20 questions have a Helpful/Not helpful rating;
- every Not-helpful question has a reviewer note;
- answers cite one of the curated technical or known-limitations sources when
  detailed behaviour is requested;
- no answer invents an unresolved interpolation or instrument-calibration
  policy;
- no answer treats convergence as proof of scientific validity;
- strain-to-stress and uncertain physical-coordinate questions receive an
  explicit scientific caution or human-review boundary.

After completing the pilot, create the local feedback summary:

```powershell
python -m tools.summarize_assistant_feedback
```

Send the printed summary path and the reviewer notes back for final refinement.
