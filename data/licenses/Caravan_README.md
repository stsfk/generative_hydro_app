# General license information

This folder contains one subdirectory per source-dataset with information on the original data license and references to the data sources. In general

- **Meteorological** time series and time series of **model states** are derived from ERA5-Land (see section below).
- **Streamflow** timeseries are taken from the various source datasets (see individual license files in the subfolder named after the source dataset for further details).
- **Catchment boundaries** are taken from the various source datasets (see individual license files in the subfolder named after the source dataset for further details).
- **Attributes**:
    - All values in files named **attributes\_hydroatlas\_\*.csv** are derived from the HydroATLAS dataset (see section below)
    - All values in files named **attributes\_caravan\_\*.csv** are derived from the processed ERA5-Land timeseries included in Caravan. 
    - For the files named **attributes\_caravan\_\*.csv**, the following columns are copied from the source datasets: `gauge_lat`, `gauge_lon`, `gauge_name`. The `area` values are inferred from the catchment polygon and the `country` were added by me (Frederik Kratzert).

**Caravan (the data) is published under the CC-BY-4.0 license. The code (under `code/` and the [GitHub repository](https://github.com/kratzert/Caravan)) is published under BSD 3-Clause.**

If you use Caravan in your research, it would be appreciated to not only cite Caravan itself, but also the source datasets, to pay respect to the amount of work that was put into the creation of these datasets and that made Caravan possible in the first place.

## ERA5-Land

Caravan contains modified Copernicus Atmosphere Monitoring Service information [2022] and neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data Caravan contains. All ERA5-Land data was processed with Google Earth Engine. ERA5-Land is published under the [Copernicus license](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf

### References

```
Muñoz Sabater, J., (2019): ERA5-Land hourly data from 1981 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). (29-Mar-2022), doi:10.24381/cds.e2161bac
```

## HydroATLAS

Caravan contains catchment attributes that are derived from the HydroATLAS dataset, which is licensed under [CC-BY-4.0](http://creativecommons.org/licenses/by/4.0/).

### References

Paper
```
Linke, S., Lehner, B., Ouellet Dallaire, C. et al. Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution. Sci Data 6, 283 (2019). https://doi.org/10.1038/s41597-019-0300-6
```
Dataset
```
Lehner, Bernhard; Linke, Simon; Thieme, Michele (2019): HydroATLAS version 1.0. figshare. Dataset. https://doi.org/10.6084/m9.figshare.9890531.v1 
```

