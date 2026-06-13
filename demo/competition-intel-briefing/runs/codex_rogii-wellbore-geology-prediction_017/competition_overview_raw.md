Drilling a horizontal well is like navigating underground without a map. The path forward runs through layers of rock you can’t see.

Build models to predict the geology along a horizontal wellbore. Your work will help automate and improve drilling operations in the oil and gas industry.

a month ago

2 months to go

### Description

Roughly 10,000 horizontal wells are drilled worldwide every year, yet much of the drilling process still relies on manual interpretation by experts. These operations require immense technical precision, where even small deviations from the target zone can lead to significant resource waste. If the well drifts into less favorable geology, it results in inefficient energy recovery and may require additional corrective measures that increase the overall environmental footprint of the site.

Interpreting the subsurface is challenging because direct measurements are inherently limited. Data from wells, seismic surveys, and logging tools only show part of the picture. Rock layers start out stacked like a layer cake, but can bend or break along faults, making it hard to know exactly where the drill bit sits within the formation. Geologists and engineers analyze incoming data to steer the well, but current analytical tools often struggle to match the nuance of expert interpretation.

In this competition, you’ll develop machine learning models that predict the geology encountered along a horizontal wellbore. Your models should identify favorable layers from drilling data and help guide well placement more accurately during operations.

Your solution could help reduce resource waste by minimizing redundant drilling, improve operational safety by better predicting geological hazards, and move the industry toward automated systems that make faster, more consistent, and data-driven decisions.

A clearer map beneath the surface could make every meter count.

### Evaluation

Submissions are scored on the root mean squared error. RMSE is defined as:

RMSE

=

1

𝑛

∑

𝑖

=

1

𝑛

(

𝑦

𝑖

−

𝑦

^

𝑖

)

2

where

𝑦

^

is the predicted value,

𝑦

is the original value, and

𝑛

is the number of rows in the test data.

### Submission File

For each row in the test set, you must predict the value of the target tvt as described on the data tab, each on a separate row in the submission file. The file should contain a header and have the following format:

id,tvt

000d7d20_1442,0.0

000d7d20_1443,0.0

000d7d20_1444,0.0

000d7d20_1445,0.0

...

### Timeline

May 5, 2026 - Start Date.

July 29, 2026 - Entry Deadline. You must accept the competition rules before this date in order to compete.

July 29, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.

August 5, 2026 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Code Requirements

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 9 hours run-time

GPU Notebook <= 9 hours run-time

Internet access disabled

Freely & publicly available external data is allowed, including pre-trained models

Submission file must be named submission.csv

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

### Prizes

1st Place - $25,000

2nd Place - $13,000

3rd Place - $7,000

4th Place - $5,000

### Citation
