The competition data comprises horizontal well trajectories and vertical reference logs (Typewells) used for geological prediction. Your goal is to predict the TVT (True Vertical Thickness) for the evaluation zone of each horizontal well.

The data is organized into train/ and test/ directories, where each well is identified by a unique 8-character hash (e.g., 015fe0d2).

File and Field Information

train/ Contains the training data. Each well has three associated files:

{WELLNAME}__horizontal_well.csv - Trajectory, geological surfaces, and log data.

WELLNAME - unique identifier for the well.

MD - Measured Depth (ft): The total length of the wellbore from the surface.

X - Easting (ft): Spatial coordinate in the horizontal plane.

Y - Northing (ft): Spatial coordinate in the horizontal plane.

Z - True Vertical Depth (ft): The vertical distance below sea level.

ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA - Predicted depth of various geological formations (Training only).

TVT - True Vertical Thickness (ft): The manually interpreted geological position for each 1 ft of the lateral well. This is the target variable (Training only).

GR - Gamma Ray (API): Log measuring natural radioactivity of the rock.

TVT_input - Input Target (ft): A copy of TVT provided as a feature. This column contains NaN values for the evaluation zone.

{WELLNAME}__typewell.csv - Vertical reference log for geological correlation.

TVT - Vertical Depth Index (ft): Primary depth reference for the vertical log. Corresponds to TVT (geological position) of the associated horizontal well.

GR - Gamma Ray (API): The vertical Gamma Ray signature used for correlation.

Geology - Formation Label: Categorical label indicating the geological unit (e.g., EGFDL, BUDA).

{WELLNAME}.png - Visualization of the well path and geological cross-section.

test/ Contains the evaluation data for about 200 wells. Each well has two associated files:

{WELLNAME}__horizontal_well.csv - Trajectory and log data. In these files, the TVT target is hidden (replaced with NaN) in the evaluation zone.

{WELLNAME}__typewell.csv - Vertical reference log for the test well.

Note that the test/ folder visible here contains only a few instances from the training set as example data to help you author your submissions. When your submission is rerun on the hidden test set, these will be replaced with the actual test data.

sample_submission.csv - A sample submission file in the correct format.

id - A unique identifier for each prediction point, formatted as {WELLNAME}_{row_index} (e.g., 015fe0d2_1654).

tvt - Your predicted True Vertical Thickness (ft).

### Files

2327 files

Size

1.33 GB

Type

csv, png, pptx

License

Subject to Competition Rules
