function saveEvents(events,filename)
%SAVEEVENTS - Save an events structure to a file
%
% Saves an events structure to a file.
%
% FUNCTION:
%   saveEvents(events,filename)
%
% INPUT ARGS:
%   events = rec_events;  % Events structure to save
%   filename = 'events/rec_events.mat'; % file to save to
%
%

%
% 2003/12/8 - PBS: Fixed that events was not a string.
%


save(filename,'events');

