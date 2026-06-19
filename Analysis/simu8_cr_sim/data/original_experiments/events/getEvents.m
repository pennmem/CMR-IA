function expr_events = getEvents(events,expr)
%GETEVENTS - Return the events that match an expression.
%
% Return the events that match an evaluated expression.  You can
% include any combination of events structure fields in your
% expression to evaluate by logical operators.  See the example
% expressions below:
%
% FUNCTION:
%   expr_events = getEvents(events,expr)
%
% INPUT ARGS:
%   events = events; % events structure to analyze
%   expr = 'rt > 1000 & strcmp(subj,''BR018'')'; % expression to eval.
%
% OUTPUT ARGS:
%   expr_events - The events matching the expression
%

% get the field names
fnames = fieldnames(events);

for f = 1:length(fnames)
  % set the expression to replace
  r_exp = ['\<' fnames{f} '\>'];
  
  % set the replacement
  r_str = ['getField(events,''' fnames{f} ''')'];
  
  % eval the expression
  expr = regexprep(expr,r_exp,r_str);
end

% get the indexes
ind = eval(['find(' expr ')']);

% return the events
expr_events = events(ind);

