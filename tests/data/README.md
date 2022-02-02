# Model Input Data #

The predictive analytics workflow includes tooling to ingest training data from
CSV files. Although CSV is not compact, it is nearly universally supported, is
human-readable, and compresses well. This README documents the default data
format expected by the workflow and describes, in brief, a synthetic test data
set included in the repository.

## Data Format ##

The data format consists of a set of 2 or more CSV files (`*.csv`) and a single
JSON configuration file (`*.json`) that describe the data set. By convention,
the CSV and JSON files should use a common, human-readable naming convention,
but this is not strictly required.

### CSV Data Files ###

Each CSV file contains a single "Timestamp" column, which by convention must be
the first column, and one or more value columns. In the current implementation,
separate CSV files are provided for target and predictor variables (which
facilitates reuse of predictor variables for many different targets). One or
more CSV files must be provided for both targets and predictors. The files must
have the following characteristics:

- The timestamp column must have the header "Timestamp" and contains datetimes
  in the ISO 8601 standard format, including time zone offset and (optionally)
  time zone name
- Value columns contain numeric data and may have arbitrary names
- Each file contains only target variables or only predictor variables
- Time ranges within each CSV file are contiguous and do not overlap with other
  CSV files of the same content type (targets or predictors)

After parsing, the range of timestamps common to all variables will be used in
model training.

### JSON Configuration File ###

The JSON configuration file captures CSV file metadata used by the data intake
pipeline to assemble the data set. For convenience of data exchange with
Haystack-compliant software, the file structure follows the Project Haystack
JSON version 3 encoding. (Note: the current Haystack JSON encoding version is 4;
this repo may be updated to version 4 encoding in the future.)

- **Lists** are encoded as JSON arrays
- **Dictionaries** are encoded as JSON objects
- **DateTimes** (timestamps) are encoded in ISO 8601 format as strings prefixed
  by "t:"
- **Strings** are encoded as strings, with or without an optional "s:" prefix
- **Numbers** are string encoded with a "n:" prefix and may include units

See the [Project Haystack documentation] for additional encoding details.

[Project Haystack documentation]: https://project-haystack.org/doc/docHaystack/Json#v3 "Project Haystack JSON Encoding"

The JSON configuration file consists of a single top-level dictionary (object)
containing five nested dictionaries (objects) that encode information about
the data set:

- `dates`: Provides the overall start and end dates (timestamps) for the data;
  this object is informational and not required for parsing
- `predictors`: Provides a list of predictor variable objects, including their
   column names (`column` field), unique identifiers if available (`id` field),
   and, optionally, other Haystack metadata
- `targets`: Provides a list of target variable objects, including at minimum
  the `column` field and following the same conventions as `predictors`
- `files`: Provides a list of all CSV files in the data set, each as an object
  with the following fields:
  - `filename`: Name of the associated CSV file
  - `contentType`: Either "targets" or "predictors"
  - `start`: Start of the time range spanned by the file
  - `end`: End of the time range spanned by the file
- `export_options`: Optional; an object with arbitrary metadata regarding how
  the data set was exported from its system of origin

## Synthetic Test Data ##

The `Synthetic Site` subdirectory of this directory contains synthetic test data
consisting of a single target variable and seven predictor variables. The data
set spans a single week (Wednesday, December 1, 2021 through Tuesday, December
7, 2021) and has the following characteristics:

- 1 minute interval
- Split into individual CSV files by day

### Predictors ###

The predictor variables represent raw weather measurements from NREL's [Solar
Radiation Research Laboratory Baseline Measurement System] (SRRL BMS) for the 
7 day period beginning December 1, 2021. The following measurements are
included:

- Dry bulb temperature
- Dew point temperature
- Relative humidity
- Global horizontal irradiance
- Direct normal irradiance
- Diffuse horizontal irradiance
- Wind speed

These measurements were intentionally selected to have nonlinear
interdependencies. (For example, global horizontal irradiance is a nonlinear
function of direct normal irradiance, diffuse horizontal irradiance, time of
day, and time of year). These nonlinearites help test the discriminatory
capabilities of machine learning algorithms.

[Solar Radiation Research Laboratory Baseline Measurement System]: https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS "SRRL BMS"

### Target ###

A single target variable is included: an artifically calculated electrical
power consumption for the hypothetical synthetic building. The target variable
is a nonlinear function of a subset of the predictor variables and of the time
of day, with built-in stochasticity. This combination of factors provides a
known, well-characterized (but still complex) behavior profile that can be used
to evaluate the goodness of fit of predictive ML algorithms. (Note, however, 
that the quantity of synthetic training data included in the repo is too small
to be used for algorithm validation.)

The target data were generated using the computed history function feature of
[SkySpark] 3.0.27. For transparency and replicability, the history function
implementation is included in the repository in `hfMachineLearningTestData.trio`.

[SkySpark]: https://skyfoundry.com/product "SkySpark software"