function dist = EDist(a,b)

%script EDist(a,b)
%
% Compute the Euclidean Distance between two vectors a and b
%
% 
% INPUT ARGS:
%    a, b 
%
% OUTPUT ARGS:
%    dist = Euclidean Distance between the two vectors

  c=a-b;
  c=c.^2;
  d=sum(c);
  d=d^.5;
  
  dist = d;
