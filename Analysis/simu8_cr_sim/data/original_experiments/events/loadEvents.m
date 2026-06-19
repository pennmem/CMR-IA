function events = loadEvents(filename)
%LOADEVENTS - Load an events structure from a file.
%
% Use this function to load an events structure from a file and set
% it to a specified variable.  The .mat file must contain a
% variable named 'events'.
%
% FUNCTION:
%   events = loadEvents(filename)
%
% INPUT ARGS:
%   filename = 'events/events.mat';  % .mat file containing the events
%
% OUTPUT ARGS:
%   events - The events structure from the file
%


load(filename)

% Wow, that was easy.



