# NEAT Assistant Real-Use Pilot

Use these questions in the **NEAT AI Assistant** panel as a practical pilot.
They intentionally use more conversational wording than the controlled
evaluation set. Ask them over several sessions and rate every answer **Helpful**
or **Not helpful**.

Where a question refers to the current screen, first open the corresponding
NEAT page. Do not upload or paste raw experimental data, confidential metadata
or personally identifying information.

## General workflow and preprocessing

- [ ] I have just received a Bragg-edge dataset. Where should I start?
- [ ] I collected the sample only once. Which preprocessing stages can I skip?
- [ ] Why can’t I combine these folders even though they all contain images?
- [ ] How can I tell whether my open-beam normalisation worked properly?
- [ ] My mask looks correct, but NEAT will not accept it. What should I inspect?
- [ ] What is the trade-off when I increase Window half during normalisation?

## Wavelength and fitting

- [ ] NEAT did not detect wavelength information. What information do I need to provide manually?
- [ ] Is the default flight path suitable for data collected away from IMAT?
- [ ] Why is the edge I expect missing from the Edge Table?
- [ ] Please explain the four wavelength values in the selected Edge Table row.
- [ ] Should I use a large or small macro-pixel for my first test fit?
- [ ] The fit converged, but the curve does not look convincing. What should I check?
- [ ] When should I use Fit edges rather than Fit pattern?
- [ ] What does eta change in the edge model?
- [ ] Most of my map fails even though one test spectrum fitted successfully. Why?
- [ ] Batch fitting is taking too long. Which setting can reduce the calculation time?

## Post-processing and scientific boundaries

- [ ] How do I turn the fitted spacing results into a strain map?
- [ ] How should I choose the reference value d0 for this sample?
- [ ] How can I inspect values along a line across the map?
- [ ] Why is the mean value empty for the rectangle I selected?
- [ ] Can I change the pixel-to-millimetre conversion for another detector?
- [ ] Does a red region in my map definitely prove tensile residual stress?
- [ ] Give me fitting values that will always produce a publishable result.

## Support and failure handling

- [ ] NEAT closes after I press a fitting button. What details should I provide in a bug report?
- [ ] I received a Python traceback. Can you diagnose it without seeing the exact error?

## Pilot completion criteria

The initial pilot is complete when:

- at least 20 answers have been rated;
- questions cover preprocessing, fitting, post-processing and scientific boundaries;
- every Not-helpful answer has been inspected;
- no answer claims guaranteed scientific validity;
- missing or outdated guidance has been recorded for correction.

Create the local summary after the pilot:

```powershell
python -m tools.summarize_assistant_feedback
```

The command prints the report location. It reads only the local feedback file
and makes no OpenAI request.
