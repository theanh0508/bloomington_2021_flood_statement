import pylab as PP
import os
import matplotlib as mpl
import collections
import datetime as dt
import netCDF4 as nc
import fastkde.fastKDE as fastKDE
import scipy.integrate
import pandas as pd
import geopandas
import xarray as xr
import schwimmbad
import cftime

import shapely.ops # for polygon unions
import shapely.geometry # for doing point-in-polygon
import shapely.prepared # for doing fast point-in-polygon
from descartes import PolygonPatch # for converting shapely objects to matplotlib PolygonPatch objects
from matplotlib.collections import PatchCollection # for efficiently drawing many polygons

# for reading text files from URLs
try:
    import urllib2 as url
except ImportError:
    import urllib.request as url
import contextlib

import numpy as np
from numpy import *


"""
Note that I got this information from the `readme.txt` file in the GHCND directory that describes the data format.
The relevant section of the `readme.txt` file is below:

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

III. FORMAT OF DATA FILES (".dly" FILES)

Each ".dly" file contains data for one station.  The name of the file
corresponds to a station's identification code.  For example, "USC00026481.dly"
contains the data for the station with the identification code USC00026481).

Each record in a file contains one month of daily data.  The variables on each
line include the following:

------------------------------
Variable   Columns   Type
------------------------------
ID            1-11   Character
YEAR         12-15   Integer
MONTH        16-17   Integer
ELEMENT      18-21   Character
VALUE1       22-26   Integer
MFLAG1       27-27   Character
QFLAG1       28-28   Character
SFLAG1       29-29   Character
VALUE2       30-34   Integer
MFLAG2       35-35   Character
QFLAG2       36-36   Character
SFLAG2       37-37   Character
  .           .          .
  .           .          .
  .           .          .
VALUE31    262-266   Integer
MFLAG31    267-267   Character
QFLAG31    268-268   Character
SFLAG31    269-269   Character
------------------------------
"""

ghcnd_station_column_format = \
"""ID            1-11   Character
LATITUDE     13-20   Real
LONGITUDE    22-30   Real
ELEVATION    32-37   Real
STATE        39-40   Character
NAME         42-71   Character
GSN_FLAG     73-75   Character
HCN/CRN_FLAG 77-79   Character
WMO_ID       81-85   Character"""

# use an ordered dictionary to store the information
# (we can use the dictionary keys later as column headers)
bounds = collections.OrderedDict()

# translate the information from the `readme.txt` table into tuples of column boundaries
# for each column
bounds['ID'] = (0,11)
bounds['YEAR'] = (11,15)
bounds['MONTH'] = (15,17)
bounds['ELEMENT'] = (17,21)
# set the widths of the VALUE*, MFLAG*, QFLAG*, and SFLAG* columns
widths = collections.OrderedDict()
widths['VALUE'] = 5
widths['MFLAG'] = 1
widths['QFLAG'] = 1
widths['SFLAG'] = 1

# set the current column index
current_index = 21
# loop through the VALUE*, MFLAG*, QFLAG*, and SFLAG* values and set the columns for each day
for i in range(31):
    for key in widths:
        bounds['{}{}'.format(key,i+1)] = (current_index,current_index + widths[key])
        current_index += widths[key]
        
# create a list of column indicies, suitable for input into pandas.read_fwf()
column_inds = [bounds[key] for key in bounds]

# create a list of the 'values columns'
value_columns = [ 'VALUE{}'.format(i) for i in range(1,32)]
# create a list of the 'QFLAG columns'
qc_columns = [ 'QFLAG{}'.format(i) for i in range(1,32)]


# create a dictionary of common variable metadata
common_variable_metadata = dict(long_name = dict(PRCP = "Precipitation",
                                                 TMAX = "Maximum Temperature",
                                                 TMIN = "Minimum Temperature",
                                                 SNOW = "Snowfall",
                                                 SNOWD = "Snow Depth",
                                                ),
                                units     = dict(PRCP = "mm",
                                                 TMAX = "C",
                                                 TMIN = "C",
                                                 SNOW = "mm",
                                                 SNOWD = "mm"
                                                ),
                                scale_factor = dict(PRCP = 0.1, TMAX = 0.1, TMIN = 0.1))

default_fill_date = dt.date(1750,1,1)
def date_or_fill(y,m,d):
    """ Construct a datetime.date object from a given year month and day.
    
        If these don't constitute a valid date, return a fill_date.
        
        """
    try:
        date = dt.date(y,m,d)
    except:
        date = default_fill_date

    return date


def read_ghcnd_variable(filename, \
                    varname = "PRCP", \
                    omit_qc_values = True, \
                    omit_missing_values = False):
    """
    Reads a GHCND '.dly' file and extracts data for a single variable.
    
    input:
    ------
        
        filename       : the path to a GHCHD '.dly' file (can be any type of path readable by pandas,
                         including a URL)
        
        varname        : the name of the variable ('ELEMENT' in the GHCND `readme.txt`) to extract
        
        omit_qc_values : flags whether to simply omit data values for which there is a quality control flag
        
        omit_missing_values : flags whether to simply omit missing values
        
        
    output:
    -------
    
        returns a tuple of numpy vectors: (data, dates, [qcflags])
        
        data : a vector of data for the given varname in the given GHCND file
        
        dates : the dates for each datum
        
        qcflags : (optional) the quality control flags for each datum; False if there is a QC issue.  
                  (Not returned if omit_qc_values is True.)
    
    """
    

    #1. Load the file with pandas

    # load the file with pandas
    ghcnd_frame = pd.read_fwf(filename,colspecs=column_inds,names=bounds.keys())
    
    #2. Subselect the desired variable and extract the column values
    
    # get the rows that have data for the desired variable
    ivar = ghcnd_frame['ELEMENT'] == varname
    
    # get a table of only the desired data values
    var_dataframe = ghcnd_frame[value_columns][ivar]
    
    #3. Convert the resulting table into a vector using numpy.ravel()
    var_data = array(var_dataframe).ravel()
    
    #4. Attempt to create a matrix of datetime objects (using 'i' as the day in 'VALUEi') for each value
    # * use try/except; it will fail for some days (e.g., April 31 and Feb 29 for non-leap years).  Return
    #   a 'fill date' for any bad dates.  This will tell us which data correspond to non-existing days.
    # * use numpy.ravel() to ravel this into a vector as well; the resulting vector should have the same
    #   length as the variable vector from step 3.
    
    # extract the year/month columns
    years = ghcnd_frame['YEAR'][ivar]
    months = ghcnd_frame['MONTH'][ivar]
    
    # create a vector of dates
    date_vector = array([ date_or_fill(y,m,d) for y,m in zip(years,months) for d in range(1,32) ])
    
    #5. Use numpy indexing to remove any bad dates from the vectors.  Either remove or set a fill value for any
    #data that have a QC flag (maybe this is a user-controlled behavior?)
    
    # get the indices of valid dates
    good_date_inds = [ i for i in range(len(date_vector)) if date_vector[i] != default_fill_date]
    # set the variable and date vectors to the subset of valid dates
    var_data = var_data[good_date_inds]
    date_vector = date_vector[good_date_inds]
    
    #5. Also extract a table of the quality-control flags.  If a QC item is present, then set the matrix to 1;
    #otherwise set it to 0.  Use numpy.ravel() to also convect this to a vector.
    qc_dataframe = ghcnd_frame[qc_columns][ivar]
    qc_vector = array(qc_dataframe).ravel()
    qc_vector = qc_vector[good_date_inds]
    # if there are missing values, then pandas sets the qc_vector type to object; so we need to
    # deal with storage of mixed data types.  I do this by treating the objects as string.
    #
    # Note also that the presence of NaN in the qc_vector indicates no value stored for the QC flag; meaning
    # that the datum at that point/time is good.
    if qc_vector.dtype == 'object':
        good_qc_inds = nonzero([ True if str(item) == 'nan' else False for item in qc_vector ])[0]
    else:
        good_qc_inds = isnan(qc_vector)
    
    if omit_qc_values:
        # omit quality control variables if flagged
        var_data = var_data[good_qc_inds]
        date_vector = date_vector[good_qc_inds]
    
    #7. Deal with missing data
    var_data = ma.masked_equal(var_data,-9999.)
    var_data = ma.masked_equal(var_data,32768.)
    var_data = ma.masked_equal(var_data,-32768.)
    date_vector = ma.masked_where(ma.getmask(var_data),date_vector)
    if omit_qc_values:
        good_qc_inds = ma.masked_where(ma.getmask(var_data),good_qc_inds)
        
    # remove missing values if requested
    if omit_missing_values:
        var_data = var_data.compressed()
        date_vector = date_vector.compressed()
        good_qc_inds = good_qc_inds.compressed()

    #8. Return the resulting vector and the comparable subset of date values
    if omit_qc_values:
        return var_data, date_vector
    else:
        return var_data, date_vector, good_qc_inds


def averageGHCNDStations( station_list, \
                          variable = 'PRCP', \
                          timeseries_start = dt.date(1850,1,1), \
                          timeseries_end = dt.date.today(), \
                          fixed_station_count = None, \
                          use_low_elevations = False, \
                          ghcnd_base = "/N/project/obrienta_startup/datasets/ghcn-daily/", \
                          elevations = None, \
                          adjusted_elevation = None,
                          lats = None, \
                          lons = None):
    """ Given a list of GHCN Daily stations, produce an average timeseries of those stations.
    
        input:
        ------
            station_list        : a list of GHCND station IDs (from the first column of the `ghcnd-stations.txt` file)
            
            variable            : the GHCND variable for which to calculate the timeseries
            
            timeseries_start    : a datetime.date object indicating the first date in the desired timeseries
            
            timeseries_end      : a datetime.date object indicating the last date in the desired timeseries
            
            ghcnd_base          : the base path in which the GHCND files reside (can be a URL or path)

            fixed_station_count : the number of stations to include in each record.  A fillvalue is provided
                                  if data are available, but not enough stations are present.  Stations
                                  in excess of the count are discarded at random if use_low_elevations is false; 
                                  otherwise, the lowest elevation stations are used. All available stations are
                                  used if None is provided (default).

            use_low_elevations  : flags how fixed_station_count handles fixed stations (see above).

            elevations, lats, lons :  if provided, the mean elevations, latitudes and/or longitudes will be calculated
                                      for each day.
                                      
            adjusted_elevation  : if provided, adiabatically adjusts the variable to the provided elevation (in meters)
                                  if not provided, no adjustment is done.  This only makes sense to do for temperature variables.

            
        output:
        -------
        
            dates, timeseries_avg, timeseries_std, count, [elevation, lat, lon]
            
            
            dates          : a list of datetime objects for the timeseries
            
            timeseries_avg : a numpy masked array of the average of the station data at each time (masked where data
                             are completely missing)
                             
            timeseries_avg : a numpy masked array of the standard deviation of the station data at each time (masked where data
                             are completely missing or where there are less than 2 stations available)
                             
            count          : the number of stations contributing to the average

            elevation      : the mean elevation of stations used in the average (only output if elevations is provided as input)

            lats           : the mean latitude of stations used in the average (only output if lats is provided as input)

            lons           : the mean longitude of stations used in the average (only output if lons is provided as input)
                             
    """
    #********************************
    # create the master set of dates
    #********************************
    # set the time units
    time_units = 'days since {:04}-{:02}-{:02} 00:00:00'.format(timeseries_start.year,timeseries_start.month,timeseries_start.day)
    # get the number of days
    num_days = (timeseries_end - timeseries_start).days
    # create the time vector
    times = arange(num_days)
    # create the dates
    dates = nc.num2date(times,time_units)
    
    # set the dry adiabatic lapse rate
    g_over_cp = -9.8076 / 1003.5 # [K/m]
    
    # check whether we need to + can do elevation adjustment
    if adjusted_elevation is not None and elevations is None:
        raise ValueError("`adjusted_elevation` is not None, but `elevations` is; elevations are required to do elevation adjustment.")
    
    
    # create a master station data array
    num_stations = len(station_list) # number of stations
    data_array = ma.masked_equal(ma.zeros([num_stations,num_days]),0)
    
    # Loop over all stations and read data
    for i in range(num_stations):
        # get the station ID
        station_id = station_list[i]
        
        # set the station path
        station_path = ghcnd_base + 'all/{}.dly'.format(station_id)

        # read the station's data
        try:
            data, datetmp = read_ghcnd_variable(station_path,variable,omit_missing_values=True)
        except:
            raise RuntimeError(print(station_path))
            
        # do the adiabatic adjustment if necessary
        if adjusted_elevation is not None:
            lapse_rate_adjustment = g_over_cp * (adjusted_elevation -  elevations[i])
            # do the adjustment assume that we need to autoscale the data before adjusting
            data = data/10 + lapse_rate_adjustment
        
        if len(data) != 0:
            # convert date objects to dates
            datetmp = [dt.datetime(d.year,d.month,d.day) for d in datetmp]

            # get the indices for the dates in the master array
            inds = array(nc.date2num(datetmp,time_units)).astype(int)

            # deal with out-of-bounds indices
            masked_inds = ma.masked_outside(inds, times[0], times[-1])

            # remove data outside the requested time
            data = ma.masked_where(ma.getmask(masked_inds),data).compressed()
            inds = masked_inds.compressed()

            # insert the data into the array
            data_array[i,inds] = data

    # deal with a fixed station count if needed
    if fixed_station_count is not None:
        # initialize the indices that will be unmasked in the final array
        unmasked_i = []
        unmasked_j = []
        # loop over all days
        for j in range(num_days):
            # get the station indices for which there are data on this day
            valid_station_inds = nonzero(logical_not(ma.getmask(data_array)[:,j]))[0]

            # if we have enough data for the day
            if len(valid_station_inds) >= fixed_station_count :

                if elevations is not None and use_low_elevations:
                    # choose from the lowest stations
                    isort = argsort(elevations[valid_station_inds])
                    for i in valid_station_inds[isort[:fixed_station_count]]:
                        unmasked_i.append(i)
                        unmasked_j.append(j)
                else:
                    # randomly choose a fixed number of stations and append the (station,day) index
                    # to the unmasking arrays
                    for i in random.choice(valid_station_inds,size=fixed_station_count,replace=False):
                        unmasked_i.append(i)
                        unmasked_j.append(j)


        # set the new data mask
        fixed_station_mask = ones(data_array.shape,dtype=bool)
        # unmask the chosen points above
        fixed_station_mask[unmasked_i,unmasked_j] = False

        # mask the data (only keeping the chosen points above)
        data_array = ma.masked_where(fixed_station_mask,data_array)

    # calculate the average and standard deviation
    data_average = ma.average(data_array,axis=0)
    data_std = ma.std(data_array,axis=0)

    # calculate the average elevation, latitude and/or longitude
    if elevations is not None:
        # create a version of elevations that is broadcast to the shape of data array
        elevationtmp = elevations[:,newaxis]*ones(shape(data_array))
        # mask the elevation array in the same places where data array is maksed
        elevationtmp = ma.masked_where(ma.getmask(data_array),elevationtmp)
        # get the average elevation at the unmasked locations
        elevation_average = ma.average(elevationtmp,axis=0)
    if lats is not None:
        # create a version of lats that is broadcast to the shape of data array
        lattmp = lats[:,newaxis]*ones(shape(data_array))
        # mask the lat array in the same places where data array is maksed
        lattmp = ma.masked_where(ma.getmask(data_array),lattmp)
        # get the average lat at the unmasked locations
        lat_average = ma.average(lattmp,axis=0)
    if lons is not None:
        # create a version of lons that is broadcast to the shape of data array
        lontmp = lons[:,newaxis]*ones(shape(data_array))
        # mask the lon array in the same places where data array is maksed
        lontmp = ma.masked_where(ma.getmask(data_array),lontmp)
        # get the average lon at the unmasked locations
        lon_average = ma.average(lontmp,axis=0)

    
    # calculate the number of stations at each time
    data_count = ma.masked_equal(sum(logical_not(ma.getmask(data_array)),axis=0),0)
    
    data_average = ma.masked_where(ma.getmask(data_count),data_average)
    data_std = ma.masked_where(logical_or(ma.getmask(data_count), data_count < 2),data_std)

    return_list = [dates, data_average, data_std, data_count]

    # append elevation/lat/lon to the list of returned variables as needed
    if elevations is not None:
        return_list.append(elevation_average)
    if lats is not None:
        return_list.append(lat_average)
    if lons is not None:
        return_list.append(lon_average)
    
    return tuple(return_list)

def __read_ghcnd_variable_wrapper__(args):
    """ Wraps read_ghcnd_variable for use with threadding"""

    i, station_id, variable, ghcnd_base, times, time_units = args

    # set the station path
    station_path = ghcnd_base + 'all/{}.dly'.format(station_id)

    # read the station's data
    try:
        data, datetmp = read_ghcnd_variable(station_path,variable,omit_missing_values=True)
    except:
        raise RuntimeError(print(station_path))

    if len(data) != 0:
        # convert date objects to dates
        datetmp = [dt.datetime(d.year,d.month,d.day) for d in datetmp]

        # get the indices for the dates in the master array
        inds = array(nc.date2num(datetmp,time_units)).astype(int)

        # deal with out-of-bounds indices
        masked_inds = ma.masked_outside(inds, times[0], times[-1])

        # remove data outside the requested time
        data = ma.masked_where(ma.getmask(masked_inds),data).compressed()
        inds = masked_inds.compressed()
    else:
        data = None
        inds = None

    return i, data, inds


def obtainGHCNDStations(  station_list, \
                          variable = 'PRCP',
                          timeseries_start = dt.date(1850,1,1),
                          timeseries_end = dt.date.today(),
                          ghcnd_base = "/N/project/obrienta_startup/datasets/ghcn-daily/",
                          elevations = None,
                          lats = None,
                          lons = None, 
                          names = None,
                          states = None,
                          num_threads = None):
    """ Given a list of GHCN Daily stations, return an xarray of timeseries data for of those stations.
    
        input:
        ------
            station_list        : a list of GHCND station IDs (from the first column of the `ghcnd-stations.txt` file)
            
            variable            : the GHCND variable for which to calculate the timeseries
            
            timeseries_start    : a datetime.date object indicating the first date in the desired timeseries
            
            timeseries_end      : a datetime.date object indicating the last date in the desired timeseries
            
            ghcnd_base          : the base path in which the GHCND files reside (can be a URL or path)

            elevations, lats, lons :  if provided, the elevations, latitudes and/or longitudes will be retained in the xarray dataset
            
            num_threads         : sets the number of threads to use for I/O.  If None is given, the default value for
                                  processes in schwimmbad.MultiPool() is used
                                      
            
        output:
        -------
        
            station_data : an xarray object with variable `variable' and dimensions [station, time], containing
                           the data from the requested variables
            
                             
    """
    #********************************
    # create the master set of dates
    #********************************
    # set the time units
    time_units = 'days since {:04}-{:02}-{:02} 00:00:00'.format(timeseries_start.year,timeseries_start.month,timeseries_start.day)
    # get the number of days
    num_days = (timeseries_end - timeseries_start).days
    # create the time vector
    times = np.arange(num_days)
    # create the dates
    dates = cftime.num2date(times,time_units)
    
    # create a master station data array
    num_stations = len(station_list) # number of stations
    data_array = np.ma.masked_equal(np.ma.zeros([num_stations,num_days]),0)
    
    threading_inputs = [ (i, station_list[i], variable, ghcnd_base, times, time_units) for i in range(num_stations) ]
    
    # read all of the data in using thread parallelism
    with schwimmbad.MultiPool(processes = num_threads) as pool:
        result = pool.map(__read_ghcnd_variable_wrapper__, threading_inputs)
        
    # Loop over all stations and read data
    for i, data, inds in result:
        if data is not None:
            # insert the data into the array
            data_array[i,inds] = data
            
    # create an xarray object from the data
    station_index = np.arange(len(station_list), dtype = int)
    var_xarray = xr.DataArray(data_array,
                              coords = dict(station = station_index,
                                            time = dates),
                              dims = ("station", "time"))
    
    # load the data into an xarray dataset
    data_xarray = xr.Dataset()
    data_xarray[variable] = var_xarray
    data_xarray['station_id'] = ("station", station_list)
    
    # add variable metadata if we have it
    if variable in common_variable_metadata['scale_factor']:
        data_xarray[variable] *= common_variable_metadata['scale_factor'][variable]
    if variable in common_variable_metadata['long_name']:
        data_xarray[variable].attrs['long_name'] = common_variable_metadata['long_name'][variable]
    if variable in common_variable_metadata['units']:
        data_xarray[variable].attrs['units'] = common_variable_metadata['units'][variable]
        
    # add the average elevation, latitude and/or longitude
    if elevations is not None:
        data_xarray['elevation'] = ('station', elevations)
        data_xarray['elevation'].attrs['long_name'] = "Station Elevation"
        data_xarray['elevation'].attrs['units'] = "m"
    if lats is not None:
        data_xarray['lat'] = ('station', lats)
        data_xarray['lat'].attrs['long_name'] = "Station Latitude"
        data_xarray['lat'].attrs['units'] = "degrees_north"
    if lons is not None:
        data_xarray['lon'] = ('station', lons)
        data_xarray['lon'].attrs['long_name'] = "Station Longitude"
        data_xarray['lon'].attrs['units'] = "degrees_east"
    if names is not None:
        data_xarray['name'] = ('station', names)
        data_xarray['name'].attrs['long_name'] = "Station Name"
    if states is not None:
        data_xarray['state'] = ('station', states)
        data_xarray['state'].attrs['long_state'] = "Station state"
        

    return data_xarray


class GHCNDAverager:

    def __init__(   self, \
                    shapely_geometry, \
                    variable = 'PRCP', \
                    timeseries_start = dt.date(1850,1,1), \
                    timeseries_end = dt.date.today(), \
                    fixed_station_count = None, \
                    use_low_elevations = False,  \
                    adjusted_elevation = None, 
                    ghcnd_base = "/N/project/obrienta_startup/datasets/ghcn-daily/"):
        """ Given a list of GHCN Daily stations, produce an average timeseries of those stations.
        
            input:
            ------
                shapely_geometry : A shapely.geometry.shape instance on a lat/lon projection
                
                variable         : the GHCND variable for which to calculate the timeseries
                
                timeseries_start : a datetime.date object indicating the first date in the desired timeseries
                
                timeseries_end   : a datetime.date object indicating the last date in the desired timeseries
                
                ghcnd_base       : the base path in which the GHCND files reside (can be a URL or path)

                fixed_station_count : the number of stations to include in each record.  A fillvalue is provided
                                      if data are available, but not enough stations are present.  Stations
                                      in excess of the count are discarded at random if use_low_elevations is false; 
                                      otherwise, the lowest elevation stations are used. All available stations are
                                      used if None is provided (default).

                use_low_elevations : flags how fixed_station_count handles fixed stations (see above).
                
                adjusted_elevation  : if provided, adiabatically adjusts the variable to the provided elevation (in meters)
                                      if not provided, no adjustment is done.  This only makes sense to do for temperature variables.

                
            saves within the object:
            ------------------------
                
                self.dates          : a list of datetime objects for the timeseries
                
                self.timeseries_avg : a numpy masked array of the average of the station data at each time (masked where data
                                      are completely missing)
                                 
                self.timeseries_std : a numpy masked array of the standard deviation of the station data at each time (masked where data
                                      are completely missing or where there are less than 2 stations available)
                                 
                self.count          : the number of stations contributing to the average

                self.stations       : the list of station IDs within the shape

                self.station_lats   : the list of station latitudes

                self.station_lons   : the list of station longitudes

                self.station_elevations : the list of station elevations

                self.average_lats   : the average latitude of stations included in the daily averages for each day
                
                self.average_lons   : the average longitude of stations included in the daily averages for each day

                self.average_elevations : the average elevation of stations included in the daily averages for each day
        """

        # """ Read in the station list """
        with open(ghcnd_base + 'ghcnd-stations.txt',encoding='utf-8') as fin:
            station_ids, lats, lons, elevations = zip(* [line.split()[:4] for line in fin.readlines()])
            
        # """ Read in the version file """
        with open(ghcnd_base + 'ghcnd-version.txt', encoding = 'utf-8') as fin:
            version_string = fin.read()
            
        # convert lat, lon, and elevation to numpy arrays
        lats = array([ float(l) for l in lats ])
        lons = array([ float(l) for l in lons ])
        elevations = array([ float(e) for e in elevations ])

        # """ Find the stations within this shape """
        stations_in_shape = []
        for i in range(len(lats)):
            # get the current station's lat/lon
            lat = lats[i]
            lon = lons[i]
            # create a shapely point out of it
            point = shapely.geometry.Point(lon,lat)
            # check if this point is within the shape
            if point.within(shapely_geometry):
                # if so, append the station ID to the list
                stations_in_shape.append(station_ids[i])

        # get the indices for this shape
        inds = nonzero([ station in stations_in_shape for station in station_ids])[0]

        # """ Do the averaging """
        dates, data_average, data_std, data_count, avg_elevations, avg_lats, avg_lons = averageGHCNDStations(stations_in_shape,
                                                                                                             variable=variable,
                                                                                                             ghcnd_base=ghcnd_base,
                                                                                                             timeseries_start = timeseries_start,
                                                                                                             timeseries_end = timeseries_end,
                                                                                                             fixed_station_count=fixed_station_count,
                                                                                                             use_low_elevations = use_low_elevations,
                                                                                                             elevations = elevations[inds],
                                                                                                             adjusted_elevation = adjusted_elevation,
                                                                                                             lats=lats[inds],
                                                                                                             lons=lons[inds])

        # store the output
        self.dates = dates
        self.timeseries_avg = data_average
        self.timeseries_std = data_std
        self.count = data_count
        self.stations = stations_in_shape
        self.station_lats = lats[inds]
        self.station_lons = lons[inds]
        self.station_elevations = elevations[inds]
        self.average_elevations = avg_elevations
        self.adjusted_elevation = adjusted_elevation
        self.average_lats = avg_lats
        self.average_lons = avg_lons
        self.version_string = version_string


def GHCNDStationsInShape(shapely_geometry,
                         variable = 'PRCP',
                         timeseries_start = dt.date(1850,1,1),
                         timeseries_end = dt.date.today(),
                         crs = {'init': 'epsg:4326'},
                         ghcnd_base = "/N/project/obrienta_startup/datasets/ghcn-daily/", 
                         ghcnd_station_index_file = None,
                         ghcnd_version_file = None,
                         num_threads = None,
                        ):
    """ Given a list of GHCN Daily stations, produce an average timeseries of those stations.

        input:
        ------
            shapely_geometry : A shapely.geometry.shape instance on a lat/lon projection

            variable         : the GHCND variable for which to calculate the timeseries

            timeseries_start : a datetime.date object indicating the first date in the desired timeseries

            timeseries_end   : a datetime.date object indicating the last date in the desired timeseries

            crs              : the crs of the shapely geometry
                               (see http://geopandas.org/projections.html)
            
            ghcnd_base       : the base path in which the GHCND files reside (can be a URL or path)
            
            ghcnd_station_index_file : the path to ghcnd-stations.txt (can be a URL or path)
                                       (default is '{ghcnd_base}/ghcnd-stations.txt')
            
            ghcnd_version_file : the path to ghcnd-versions.txt (can be a URL or path)
                                 (default is '{ghcnd_base}/ghcnd-versions.txt')
                                 NOTE: it is recommended that the default value not be overriden,
                                 otherwise there is a risk that the ghcnd-version stored in the
                                 `station_data` metadata will not match the actual ghcnd-version
                                 of the data read in.

            num_threads       : sets the number of threads to use for I/O.  If None is given, the default value for
                                processes in schwimmbad.MultiPool() is used
        output:
        -------
        
            station_data : an xarray object with variable `variable' and dimensions [station, time], containing
                           the data from the requested variables and metadata about station lat, lon and elevation
    """
    # set default paths
    if ghcnd_station_index_file is None:
        ghcnd_station_index_file = ghcnd_base + '/ghcnd-stations.txt'
    if ghcnd_version_file is None:
        ghcnd_version_file = ghcnd_base + '/ghcnd-version.txt'

    # """ Read in the station list """
        
    # map the types in the station file to Python types
    type_mapping = {"Real" : float, "Character" : str }

    # convert the station information into headers and fixed-width format specifiers expected by pandas.read_fwf()
    names = []
    colspecs = []
    dtypes = []
    for line in ghcnd_station_column_format.split('\n'):
        name, columns, ftype = line.split()
        # update the header
        # replace _ with blank space (I manually changed spaces in names to aid with parsing)
        names.append(name.replace('_',' '))

        # update the fixed-width specifier
        start,end = columns.split('-')
        colspecs.append((int(start)-1,int(end)))

        # update the data type
        dtypes.append(type_mapping[ftype])

    # read the station index
    stations_pd = pd.read_fwf(ghcnd_station_index_file, names = names, colspecs=colspecs, dtypes=dtypes)
    
    """ Convert the pandas dataframe into a geopandas dataframe """
    # create shapely points for the lat/lon values for each station
    geometry = [shapely.geometry.Point(xy) for xy in zip(stations_pd['LONGITUDE'], stations_pd['LATITUDE'])]

    # create a geopandas dataframe out of the pandas dataframe
    # set the CRS to the one used by the CA HUC shapefile
    stations_pd = geopandas.GeoDataFrame(stations_pd, crs=crs, geometry=geometry)


    # """ Read in the version file """
    try:
        with contextlib.closing(url.urlopen(ghcnd_version_file)) as fin:
            version_string = fin.read().decode('utf-8')
    except ValueError:
        with open(ghcnd_version_file, encoding = 'utf-8') as fin:
            version_string = fin.read()

    # """ Find the stations within the given shape """
    # create a prepared-geometry file for fast point-in-polygon searching
    shapely_geometry_prep  = shapely.prepared.prep(shapely_geometry)

    # make points into Shapely point object
    station_in_shape = [shapely_geometry_prep.intersects(point) for point in stations_pd.geometry]
    stations_in_shape_pd = stations_pd[station_in_shape]
    
    # get the station data
    station_data = obtainGHCNDStations(list(stations_in_shape_pd['ID']),
                                       variable=variable,
                                       ghcnd_base=ghcnd_base,
                                       timeseries_start = timeseries_start,
                                       timeseries_end = timeseries_end,
                                       elevations = stations_in_shape_pd['ELEVATION'].values,
                                       lats = stations_in_shape_pd['LATITUDE'].values,
                                       lons = stations_in_shape_pd['LONGITUDE'].values,
                                       names = stations_in_shape_pd['NAME'].values,
                                       states = stations_in_shape_pd['STATE'].values,
                                       num_threads = num_threads)


    # store the GHCND version information
    station_data.attrs["ghcnd_version_info"] = version_string
    
    return station_data
