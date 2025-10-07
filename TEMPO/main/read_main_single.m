function [savename] = read_main_single(struct_location, struct_name, varargin)
% READ_MAIN_SINGLE Reads in preprared NO2, MODIS, and GLOBE data to .mat files
%
%   READ_MAIN is the first step in the BEHR workflow. It reads
%   in the satellite data from the various sources, include OMI NO2, MODIS
%   clouds, MODIS albedo, and GLOBE (a database, not a satellite) terrain
%   elevation. These data are cut down to the US domain and, in the case of
%   the MODIS and GLOBE data, averaged to the OMI pixels. The resulting
%   Data structures are saved as an OMI_SP .mat file.
%
%   This function is setup such that running it without arguments will
%   produce any new OMI_SP files required. This requires that the necessary
%   data be available either locally or via a mounted network drive. This
%   behavior can be changed with the following parameters:
%
%       'sp_mat_dir' - the directory that the OMI_SP .mat files will be
%       saved to. Default is the path provided by the behr_paths class.
%
%       'modis_myd06_dir' - the directory that contains the MODIS MYD06
%       cloud HDF4 files, sorted by year. Default is the path provided by
%       the behr_paths class.
%
%       'modis_mcd43_dir' - the directory that contains the MODIS MCD43C1
%       BRDF parameters files, sorted into subdirectories by year. Default
%       is the path provided by the behr_paths class.
%
%       'modis_land_mask_path' 
%
%       'globe_dir' - the directory that contains the GLOBE (Global Land
%       One-km Base Elevation) terrain elevation data. This will contain
%       files a10g through p10g and a10g.hdr through p10g.hdr. Default is
%       the path provided by the behr_paths class.
%
%       'region' - which region BEHR is running in. This controls both the
%       longitude and latitude limits and which orbits are skipped as
%       "nighttime" orbits. This must be a string. Default (and only option
%       at present) is 'US'.
%
%       'allow_no_myd' - boolean (default false) which allows the run to
%       process days for which no MODIS cloud fraction data is available.
%
%       'overwrite' - scalar logical which controls whether existing files
%       will be overwritten. If false, a day will be skipped if the
%       corresponding OMI_SP .mat file exists in the directory given as
%       'omi_he5_dir'. If true, no days will be skipped and the data in
%       omi_he5_dir will be overwritten.
%
%       'DEBUG_LEVEL' - verbosity. Default is 2; i.e. most progress
%       message, but no timing messages will be printed. 0 = no messages;
%       greater means more messages.
%
%       'save_as_pydict' - scalar logical which determines if the output
%       structure is save as a pickled Python dictionary or as a MATLAB
%       structure. If true (default), the structure is converted to a
%       Python dictionary and pickled at sp_mat_dir. If false, the 
%       structure is directly saved to sp_mat_dir as a .mat file.
%
%       'lonmin' - if 'region' is set to 'custom', specifies the minimum
%       longitude of region

%       'lonmax' - if 'region' is set to 'custom', specifies the maximum
%       longitude of region

%       'latmin' - if 'region' is set to 'custom', specifies the minimum
%       latitude of region

%       'latmax' - if 'region' is set to 'custom', specifies the maximum
%       latitude of region
%
%       'albedo_only' - defaults to true. If true, only the MODIS albedo
%       values are added; MODIS cloud and GLOBE terrain are ignored.

%****************************%
% CONSOLE OUTPUT LEVEL - 0 = none, 1 = minimal, 2 = all messages, 3 = times

%****************************%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% INITIALIZATION & INPUT VALIDATION %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try
    E = JLLErrors;
catch 
    warning('JLLErrors was not recognized. Using JLLErrorsLocal.')
    E = JLLErrorsLocal;
end

p = inputParser;
p.addParameter('sp_mat_dir', '');
p.addParameter('modis_myd06_dir', '');
p.addParameter('modis_mcd43_dir', '');
p.addParameter('modis_land_mask_path', '');
p.addParameter('globe_dir', '');
p.addParameter('region', 'US');
p.addParameter('allow_no_myd', false);
p.addParameter('overwrite', false)
p.addParameter('DEBUG_LEVEL', 2);
p.addParameter('read_from_pydict', true);
p.addParameter('save_as_pydict', true);
p.addParameter('behr_path', '');
p.addParameter('lonmin', -125);
p.addParameter('lonmax', -65);
p.addParameter('latmin', 25);
p.addParameter('latmax', 50);
p.addParameter('albedo_only', '1');

p.parse(varargin{:});
pout = p.Results;

sp_mat_dir = pout.sp_mat_dir;
modis_myd06_dir = pout.modis_myd06_dir;
modis_mcd43_dir = pout.modis_mcd43_dir;
modis_land_mask_path = pout.modis_land_mask_path;
globe_dir = pout.globe_dir;
allow_no_myd = pout.allow_no_myd;
region = pout.region; 
overwrite = pout.overwrite;
DEBUG_LEVEL = pout.DEBUG_LEVEL;
read_from_pydict = pout.read_from_pydict;
save_as_pydict = pout.save_as_pydict;
behr_path = pout.behr_path;
albedo_only = pout.albedo_only;

albedo_only = int32(str2double(albedo_only));

if albedo_only
    fprintf('Will only add MODISAlbedo variables\n')
else
    fprintf('Will add MODIS and GLOBE variables\n')
end

if ~isempty(behr_path)
    % Use the provided directory to add to the path
    s = genpath(behr_path);
    addpath(s)
end

%%%%%%%%%%%%%%%%%%%%%%
%%%%% VALIDATION %%%%%
%%%%%%%%%%%%%%%%%%%%%%

if ~ischar(sp_mat_dir)
    E.badinput('Paramter "sp_mat_dir" must be a string')
elseif ~ischar(modis_myd06_dir)
    E.badinput('Paramter "modis_myd06_dir" must be a string')
elseif ~ischar(modis_mcd43_dir)
    E.badinput('Paramter "modis_mcd43_dir" must be a string')
elseif ~ischar(globe_dir)
    E.badinput('Paramter "globe_dir" must be a string')
elseif ~ischar(region)
    E.badinput('Paramter "region" must be a string')
elseif (~islogical(overwrite) && ~isnumeric(overwrite)) || ~isscalar(overwrite)
    E.badinput('Parameter "overwrite" must be a scalar logical or number')
end

% Default ancillary buffer for latlim and lonlim is 10 degrees
ancillary_buffer = 10;

% Specify the longitude and latitude ranges of interest for this retrieval.
% Additionally, set the earliest and latest start time (in UTC) for the
% swaths that will be allowed. This will help 
switch lower(region)
    case 'us'
        lonmin = -125;    
        lonmax = -65;
        latmin = 25;    
        latmax = 50;
    case 'hk'
        lonmin = 108;
        lonmax = 118;
        latmin = 19;
        latmax = 26;
    case 'custom'
        fprintf('Running in custom region mode:\n')
  
        lonmin = pout.lonmin;
        lonmax = pout.lonmax;
        latmin = pout.latmin;
        latmax = pout.latmax;

        if ischar(pout.lonmin) | isstring(pout.lonmin)
            % If one is a string, assume the rest are strings
            lonmin = str2double(lonmin);
            lonmax = str2double(lonmax);
            latmin = str2double(latmin);
            latmax = str2double(latmax);
        end

        % Change ancillary buffer to help in small running mode
        ancillary_buffer = 1;

        fprintf('lonmin: %.2f\n', lonmin);
        fprintf('lonmax: %.2f\n', lonmax);
        fprintf('latmin: %.2f\n', latmin);
        fprintf('latmax: %.2f\n', latmax);
        warning('Using ancillary_buffer = %i. Buffer of 10 is recommended.\n', ancillary_buffer);
    otherwise 
        E.badinput('Region "%s" not recognized', region)
end

if lonmin > lonmax %Just in case I enter something backwards...
    E.callError('bounds', 'Lonmin is greater than lonmax')
elseif latmin > latmax
    E.callError('bounds', 'Latmin is greater than latmax')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% DATA DIRECTORIES %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% These are the directories to save or load data from. By default, they are
% taken from the behr_paths class, which can be created by running
% BEHR_initial_setup in the root folder of this repo. Alternately, paths
% can be specified as parameter inputs to this function (useful on the
% cluster where this should be called from a run script or in unit tests
% where the default save directories need to be overwritten).

%This is the directory where the final .mat file will be saved.
if isempty(sp_mat_dir)
    sp_mat_dir = behr_paths.SPMatSubdir(region);
end

fprintf('Files will be saved at %s\n', sp_mat_dir);

if ~albedo_only
    %This is the directory where the MODIS myd06_L2*.hdf files are saved.
    %It should include subfolders organized by year.
    if isempty(modis_myd06_dir)
        modis_myd06_dir = behr_paths.myd06_dir;
    end
end

%This is the directory where the MODIS MCD43C3*.hdf files are saved. It
%should include subfolders organized by year.
if isempty(modis_mcd43_dir)
    modis_mcd43_dir = behr_paths.mcd43d_dir;
end

if ~albedo_only
    %This is the directory where the GLOBE data files and their headers
    %(.hdr files) are saved.
    if isempty(globe_dir)
        globe_dir = behr_paths.globe_dir;
    end
end

% Verify the paths integrity.
nonexistant = {};

if ~exist(sp_mat_dir,'dir')
    nonexistant{end+1} = 'sp_mat_dir';
end
if ~exist(modis_mcd43_dir,'dir')
    nonexistant{end+1} = 'modis_mcd43_dir';
end
if ~albedo_only
    if ~exist(modis_myd06_dir,'dir')
        nonexistant{end+1} = 'modis_myd06_dir';
    end
    if ~exist(globe_dir,'dir')
        nonexistant{end+1} = 'globe_dir';
    end
end


if numel(nonexistant)>0
    string_spec = [repmat('\n\t%s',1,numel(nonexistant)),'\n\n'];
    msg = sprintf('The following paths are not valid: %s Please double check them in the run file',string_spec);
    E.callError('bad_paths',sprintf(msg,nonexistant{:}));
end

%%%%%%%%%%%%%%%%%%%%%
%%%%% MAIN BODY %%%%%
%%%%%%%%%%%%%%%%%%%%%

% Set instrument as TEMPO
instrument = 'TEMPO';

% Make sure struct_name is a string before continuing
struct_name = convertCharsToStrings(struct_name);

% Extract the measurement date from the file name
this_datetime = datetime(extractBetween(struct_name, 18, 32)); % slice is given in the TROPOMI PUM
% Convert to datenum (no longer recommended)
this_dnum = datenum(this_datetime);

% Based on an examination of adjacent orbit, the start and times of the
% granule (in the TROPOMI file name) are equivalent to the swath starttime
% and the next swath starttime
if ~albedo_only
    tropomi_starttime = floor(double(extractBetween(struct_name, 30, 35)) / 100);
    tropomi_next_starttime = floor(double(extractBetween(struct_name, 46, 51)) / 100);
end
tempo_granule = extractBetween(struct_name, 40, 41);
tempo_scan = extractBetween(struct_name, 36, 38);

%Add a little buffer around the edges to make sure we have ancillary data
%everywhere that we have NO2 profiles.
ancillary_lonlim = [lonmin - ancillary_buffer, lonmax + ancillary_buffer];
ancillary_latlim = [latmin - ancillary_buffer, latmax + ancillary_buffer];

%Load the land classification map. We'll use this to decide when to use the
%ocean surface reflectance parameterization. 
if DEBUG_LEVEL > 1; fprintf('Loading land/ocean classification map\n'); end
if DEBUG_LEVEL > 2; t_load_land_ocean = tic; end

[ocean_mask.mask, ocean_mask.lon, ocean_mask.lat] = get_modis_ocean_mask(ancillary_lonlim, ancillary_latlim);

if DEBUG_LEVEL > 2; fprintf('    Time to load land/ocean classification map: %f\n', toc(t_load_land_ocean)); end

if ~albedo_only
    %Go ahead and load the terrain pressure data - only need to do this once
    if DEBUG_LEVEL > 1; fprintf('Loading globe elevations\n'); end
    if DEBUG_LEVEL > 2; t_load_globe = tic; end
    
    [globe_elevations, globe_lon_matrix, globe_lat_matrix] = load_globe_alts(ancillary_lonlim, ancillary_latlim, 'vector');
    globe_elevations(isnan(globe_elevations)) = 0;
    if DEBUG_LEVEL > 2; fprintf('    Time to load GLOBE elevations: %f\n', toc(t_load_globe)); end
end

if DEBUG_LEVEL > 1; fprintf('Loading COART sea reflectances\n'); end
if DEBUG_LEVEL > 2; t_load_coart = tic; end
[~, coart_lut] = coart_sea_reflectance(0);
if DEBUG_LEVEL > 2; fprintf('    Time to load COART look up table: %f\n', toc(t_load_coart)); end

% Setup some values that either need to be computed to determine the loop
% indices or which are better calculated outside the loop.

core_githead = git_head_hash(behr_paths.behr_core);
behrutils_githead = git_head_hash(behr_paths.behr_utils);
genutils_githead = git_head_hash(behr_paths.utils);
behr_grid = GlobeGrid(0.05, 'domain', [lonmin, lonmax, latmin, latmax]);

this_year = year(this_dnum);
this_year_str = sprintf('%04d', this_year);
this_month=month(this_dnum);
this_month_str = sprintf('%02d', this_month);
this_day=day(this_dnum);
this_day_str = sprintf('%02d', this_day);

% SB (2025-02-02): Commenting this block out because it doesn't work as
% written when we are timestamping files at the point of generation. To
% restore functionality, this should check only part of the file name.
%
% Check if the file already exists. If it does, and if we're set
% to not overwrite, we don't need to process this day.
%savename = sp_savename(this_dnum, region, instrument);
%if exist(fullfile(sp_mat_dir, savename), 'file') && ~overwrite
%    if DEBUG_LEVEL > 0; fprintf('File %s exists, skipping this day\n', savename); end
%end

% Load the structure we prepared with Python
if read_from_pydict
    handle = py.open(strcat(struct_location, '/', struct_name), 'rb');
    py_dict = py.pickle.load(handle);
    this_data = pydict2struct(py_dict);
    handle.close()
    clear py_dict
else 
    this_data = load(strcat(struct_location, '/', struct_name));
    this_data = this_data.pdict;
end

if ~albedo_only
    % Add MODIS cloud info to the files 
    if DEBUG_LEVEL > 0; fprintf('\n Adding MODIS cloud data \n'); end
    
    if DEBUG_LEVEL > 2; t_modis_cld = tic; end
    this_data = read_modis_cloud(modis_myd06_dir, this_dnum, this_data, tropomi_starttime, tropomi_next_starttime, [lonmin, lonmax], [latmin, latmax],...
        'AllowNoFile', allow_no_myd, 'DEBUG_LEVEL', DEBUG_LEVEL, 'LoncornField', 'TiledCornerLongitude', 'LatcornField', 'TiledCornerLatitude');
    if DEBUG_LEVEL > 2; fprintf('      Time to average MODIS clouds on worker %d: %f\n', this_task.ID, toc(t_modis_cld)); end
end

% Add MODIS albedo info to the files
if DEBUG_LEVEL > 0; fprintf('\n Adding MODIS albedo information \n'); end
if DEBUG_LEVEL > 2; t_modis_alb = tic; end

% Changed from FoV75 to tiled corners. TROPOMI only has tiled corners
[orbit_lonlim, orbit_latlim] = calc_orbit_latlon_limis(this_data.TiledCornerLongitude, this_data.TiledCornerLatitude, ancillary_lonlim, ancillary_latlim);

% Previously we tried doing this outside the orbit loop, which used
% a lot of memory but limited the number of times that we had to
% read these files. Now, we'll try it inside the loop, but only
% read the part relevant to each orbit.
if DEBUG_LEVEL > 2; t_alb_read = tic; end
modis_brdf_data = read_modis_albedo(modis_mcd43_dir, this_dnum, orbit_lonlim, orbit_latlim);
if DEBUG_LEVEL > 2; fprintf('Worker %d: Time to read MODIS BRDF = %f\n', this_task.ID, toc(t_alb_read)); end

this_data = avg_modis_alb_to_pixels(modis_brdf_data, coart_lut, ocean_mask, this_data, 'QualityLimit', 3, 'DEBUG_LEVEL', DEBUG_LEVEL,...
    'LoncornField', 'TiledCornerLongitude', 'LatcornField', 'TiledCornerLatitude');

if DEBUG_LEVEL > 2; fprintf('      Time to average MODIS albedo on worker %d: %f\n', this_task.ID, toc(t_modis_alb)); end

if ~albedo_only
    % Add GLOBE terrain pressure to the files
    
    % SB 2025-02-06: Added functionality to limit GLOBE averaging to a
    % selection of scanline and ground-pixel values
    MirrorStepOffset = this_data.mirror_step(1);
    MirrorStepMin = this_data.MirrorStepBdy(1) - MirrorStepOffset;
    MirrorStepMax = this_data.MirrorStepBdy(2) - MirrorStepOffset;

    XTrackOffset = this_data.xtrack(1);
    XTrackMin = this_data.XTrackBdy(1) - XTrackOffset;
    XTrackMax = this_data.XTrackBdy(2) - XTrackOffset;
    
    if DEBUG_LEVEL > 0; fprintf('\n Adding GLOBE terrain data \n'); end
    if DEBUG_LEVEL > 2; t_globe = tic; end
    this_data = avg_globe_data_to_pixels(this_data, globe_elevations, globe_lon_matrix, globe_lat_matrix,...
        'DEBUG_LEVEL', DEBUG_LEVEL, 'LoncornField', 'TiledCornerLongitude', 'LatcornField', 'TiledCornerLatitude',...
        'ScanlineMin', MirrorStepMin, 'ScanlineMax', MirrorStepMax, 'GroundPixelMin', XTrackMin,...
        'GroundPixelMax', XTrackMax);
    if DEBUG_LEVEL > 2; fprintf('      Time to average GLOBE data on worker %d: %f\n', this_task.ID, toc(t_globe)); end
end

% Add the few attribute-like variables
this_data.Date = datestr(this_dnum, 'yyyy/mm/dd');
this_data.LonBdy = [lonmin, lonmax];
this_data.LatBdy = [latmin, latmax];
%this_data.GitHead_Core_Read = core_githead;
%this_data.GitHead_BEHRUtils_Read = behrutils_githead;
%this_data.GitHead_GenUtils_Read = genutils_githead;
%this_data.Grid = behr_grid;
this_data.BEHRRegion = lower(region);

% Clear the modis albedo structure, hopefully this will help with
% memory usage
modis_brdf_data = [];

% Save the data
% If we want to save as a python dictionary, first convert to a dictionary
% then store as a pickle file
if save_as_pydict
    %Data = this_data.Data;
    % Remove the 'Grid' field because struct2pydict doesn't know an
    % equivalent variable type in python
    %Data = rmfield(Data, 'Grid');
    % Convert to dictionary
    this_pydict = struct2pydict( this_data );
    % Change savename to have the extension .pickle
    savename = sp_savename_tempo(this_dnum, region, instrument, '.pickle', 'REDv0-1', tempo_scan, tempo_granule);
    python_make_pickle( this_pydict , fullfile(sp_mat_dir, savename) )
    % 2024-10-15: This should output a dictionary with only one layer,
    % where the keys are the variables

% Otherwise, save as a .mat structure
else
    saveData(fullfile(sp_mat_dir,savename), this_data);
end

if DEBUG_LEVEL > 2; fprintf('    Time for one orbit: %f\n', toc(t_orbit)); end

end

function saveData(filename,Data)
save(filename,'Data')
end

function mycleanup()
err=lasterror;
if ~isempty(err.message)
    fprintf('MATLAB exiting due to problem: %s\n', err.message);
    if ~isempty(gcp('nocreate'))
        delete(gcp)
    end
    
    exit(1)
end
end

function data = handle_corner_zeros(data, DEBUG_LEVEL)
fns = fieldnames(data);
ff = ~iscellcontents(regexpi(fns, 'corner', 'once'), 'isempty');
fns = fns(ff);
for a=1:numel(fns)
    xx = all(data.(fns{a}) == 0, 1);
    if any(xx(:)) && DEBUG_LEVEL > 0
        fprintf('    Pixels with all corners == 0 found for field %s, setting corners to NaN\n', fns{a});
    end
    data.(fns{a})(:,xx) = nan;
end
end

function [lonlim, latlim] = calc_orbit_latlon_limis(lons, lats, anc_lonlim, anc_latlim)
% Figure out the lat/lon extents of the orbit, with a small buffer. Right
% now, this is just used for the MODIS BRDF data, which is at 30 arc sec
% (i.e. 1/120 of a degree or ~ 1 km) resolution, so we'll add about two
% grid cells in each direction. Also restrict it to the ancillary data
% limits so that it is consistent with ancillary data loaded for the whole
% day.
buffer = 2/120;
lonlim = [min(lons(:))-buffer, max(lons(:))+buffer];
lonlim = [max(lonlim(1), anc_lonlim(1)), min(lonlim(2), anc_lonlim(2))];

latlim = [min(lats(:))-buffer, max(lats(:))+buffer];
latlim = [max(latlim(1), anc_latlim(1)), min(latlim(2), anc_latlim(2))];
end
