function events = loadSubjEvents(basedir,subj,eventfile)
%LOADSUBJEVENTS - Combine a number of subject's events
%
% Given a base directory and list off subjects, this function will
% loop through and load each subject's events and concatenate them 
% into a single events structure.
%
% FUNCTION:
%   events = loadSubjEvents(basedir,subj,eventfile)
%
% INPUT ARGS:
%   basedir = '~/eeg/free';   % Root directory to look for files
%   subj = {'CH003','CH005'}; % Cell array of subjects to load
%   eventfile = 'events/events.mat';  % directory and file to load
%
% OUTPUT ARGS:
%   events- Events struture of all combined events
%

% save starting dir
startdir = pwd;

events = [];

for s = 1:length(subj)
  % go to 
  subjdir = fullfile(basedir,subj{s});
  cd(subjdir)
  
  % load the events
  new_events = loadEvents(eventfile);
  
  % check the fieldnames
  nfn = fieldnames(new_events);
  if ~isempty(events)
    cfn = fieldnames(events);
  else
    cfn = {};
  end
  
  if length(nfn)==length(cfn) | isempty(cfn)
    % load them cause should match
    events = [events , loadEvents(eventfile)];
  else
    % add dummy fields
  end
  
end

% return to starting dir
cd(startdir)

