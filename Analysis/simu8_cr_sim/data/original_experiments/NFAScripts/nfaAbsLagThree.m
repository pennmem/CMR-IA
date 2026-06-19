function [pconf, pconf1, pconf2, b, b1, b2] = nfaAbsLag(events)

% FUNCTION pconf = nfaAbsLag(events)
%  Finds all of the possible errors that could be made, calculates 
%  the temporal lags at study of those errors, then finds the 
%  probability based on the distances taht a confusion will be made
% 
% Input Arg:
%  events=events:structure you want to analyze
% Output Arg:
%  pconf=matrix of probabilities of making an error based on lag distance

  allevents=events;
  allevents=filterStruct(allevents,'blocknum==1 | blocknum==11');
  
  
  for e=1:length(allevents)
     allevents(e).lag=[];
  end
  
  allevents1=filterStruct(allevents,'blocknum==1');
  allevents2=filterStruct(allevents,'blocknum==11');

  pconf=LagSubFunctionThree(allevents);
  pconf1=LagSubFunctionThree(allevents1);
  pconf2=LagSubFunctionThree(allevents2);
 
  x = [1 2 3];
  
  p = polyfit(x, pconf, 1); 
  p1 = polyfit(x, pconf1, 1);
  p2 = polyfit(x, pconf2, 1);

  b=p(1);
  b1=p1(1);
  b2=p2(1); 

   
  
