function pconf = nfaTheorSim(events)

% FUNCTION nfaTheorSim(events)
%  Finds all of the possible errors that could be made, calculates the theoretical distances of those errors, then finds the probability based on the distances taht a confusion will be made
% 
% Input Arg:
%  events=events:structure you want to analyze
%  


  % filter the structure so that it only includes 'test' items
  allevents=events;
  events=filterStruct(allevents, 'typeevent==2');
            
  % find all the possible neighbors and the confusiosn and from taht find the probailitiy of confusion based on similarity
  neigh=[0 0 0 0 0 0 0 0 0 0];
  conf=[0 0 0 0 0 0 0 0 0 0];
  pconf=[];
  i=1;
  while i<=length(events)
     for j=1:length(events)
        if strcmp(events(i).subject,events(j).subject)==1 & events(i).halfnum==events(j).halfnum & events(i).blocknum==events(j).blocknum & events(i).face~=events(j).face
           %if EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>0 & EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)<=.5
         	%neigh(1)=neigh(1)+1;
           %elseif EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>.5 & EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)<=1.0
                %neigh(2)=neigh(2)+1;
	   %elseif EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>1 & EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)<=1.5
	        %neigh(3)=neigh(3)+1;
	   if EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>1.5 & EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)<=2
	        neigh(4)=neigh(4)+1;
	   elseif EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>2 & EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)<=2.5
	        neigh(5)=neigh(5)+1;
	   elseif EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>2.5 & EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)<=3
	        neigh(6)=neigh(6)+1;
	   elseif EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>3 & EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)<=3.5
	        neigh(7)=neigh(7)+1;
	   elseif EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>3.5 & EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)<=4
	        neigh(8)=neigh(8)+1;
	   %elseif EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>4 & EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)<=4.5
	       %neigh(9)=neigh(9)+1;
           %elseif EDist(events(i).theorfacecoordinates,events(j).theorfacecoordinates)>4.5
		%neigh(10)=neigh(10)+1;
           end
        end
     end
  i=i+1;
  end
  
  % Filter the structure so that only 'recall' events are shown and there are no confusions or vocalizations, then find the confusions for every distance bin
  events = filterStruct(allevents, 'typeevent==3');
  events = filterStruct(events, 'intrusion==0 & vocalization==0'); 
  
  i=1;
  while i<=length(events)
    % if events(i).theordistance>0 & events(i).theordistance<=.5
    %    conf(1)=conf(1)+1;
    % elseif events(i).theordistance>.5 & events(i).theordistance<=1
    %    conf(2)=conf(2)+1;
    % elseif events(i).theordistance>1 & events(i).theordistance<=1.5
    %    conf(3)=conf(3)+1;
     if events(i).theordistance>1.5 & events(i).theordistance<=2
        conf(4)=conf(4)+1;
     elseif events(i).theordistance>2 & events(i).theordistance<=2.5
        conf(5)=conf(5)+1;
     elseif events(i).theordistance>2.5 & events(i).theordistance<=3
        conf(6)=conf(6)+1;
     elseif events(i).theordistance>3 & events(i).theordistance<=3.5
        conf(7)=conf(7)+1;
     elseif events(i).theordistance>3.5 & events(i).theordistance<=4
        conf(8)=conf(8)+1;
    % elseif events(i).theordistance>4.0 & events(i).theordistance<=4.5
    %    conf(9)=conf(9)+1; 
    % elseif events(i).theordistance>4.5
    %    conf(10)=conf(10)+1;
     end
     i=i+1;
  end
  
  % now find the probability, and see which have non-zero values
  pconf=[conf(1)/neigh(1) conf(2)/neigh(2) conf(3)/neigh(3) conf(4)/neigh(4) conf(5)/neigh(5) conf(6)/neigh(6) conf(7)/neigh(7) conf(8)/neigh(8)] %conf(9)/neigh(9) conf(10)/neigh(10)]
  % it looks like there are no neighbors between 0 and .5 distance, so lets make the array smaller
  pconf=[pconf(4) pconf(6) pconf(7) pconf(8)];
  
  % Now, lets plot that mofo  
  x=[2 3 3.5 4];
  y=pconf;
 % plot(x,y,'r-', 'LineWidth',3)
 % publishfig
 % xlabel('Theoretical Distance Bins')
 % ylabel('P(Confusion)')
 % title('Confusability as a Function of Theoretical Distance')  
    
  pconf = pconf;
