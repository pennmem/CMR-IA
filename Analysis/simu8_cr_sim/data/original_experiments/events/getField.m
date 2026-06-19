function data = getField(events,field,expr);
%GETFIELD - Return data from a field in an events structure.
%
% Return all the data from a single field in an events structure.
% You can then use the returned data in logical comparisons to
% filter the events.
%
% FUNCTION:
%   data = getField(events,field,expr)
%
% INPUT ARGS:
%   events = events;   % Events structure to query
%   field = 'fileno';  % Field in events structure to query
%   expr = 'rt > 1000 & strcmp(subj,''BR018'')' % Optional expression to
%                                              % limit events. (Defaults to '')
%
% OUTPUT ARGS:
%   data- A vector or Cell array (If the field contained strings)
%   of the desired field's data.
%

% set default arg
if nargin < 3
  expr = '';
end

% see limit events first
if length(expr) > 0
  events = getEvents(events,expr);
end

% see if contains character data
if length(events)==0
  x = [];
elseif eval(['isempty(events(1).' field ')']) | eval(['ischar(events(1).' field ')'])
  % it's a char or null so just copy it staight
  x = eval(['{events.' field '}']);
else  
  % it is numeric data, so get it with []
  x = eval(['[events.' field ']']);  
end

% return the data
data = x;

