function pconf = nfaSimilarity2(events)

% FUNCTION nfaSimilarity2(events)
%  Finds all of the possible errors that could be made, calculates the distances of those errors, then finds the probability based on the distances taht a confusion will be made
% 
% Input Arg:
%  events=events:structure you want to analyze
%  


  % filter the structure so that it only includes 'test' items
  allevents=events;
  events=filterStruct(allevents, 'typeevent==2');
            
  % find all the possible neighbors and the confusiosn and from taht find the probailitiy of confusion based on similarity
  neigh=[0 0 0 0 0 0 0 0 0];
  conf=[0 0 0 0 0 0 0 0 0];
  pconf=[];
  i=1;
  while i<=length(events)
     for j=1:length(events)
        if strcmp(events(i).subject,events(j).subject)==1 & events(i).halfnum==events(j).halfnum & events(i).blocknum==events(j).blocknum & events(i).face~=events(j).face
           if EDist(events(i).facecoordinates,events(j).facecoordinates)>0 & EDist(events(i).facecoordinates,events(j).facecoordinates)<=1.5
                neigh(1)=neigh(1)+1;
	   elseif EDist(events(i).facecoordinates,events(j).facecoordinates)>1.5 & EDist(events(i).facecoordinates,events(j).facecoordinates)<=2.5
	        neigh(2)=neigh(2)+1;
	   elseif EDist(events(i).facecoordinates,events(j).facecoordinates)>2.5 & EDist(events(i).facecoordinates,events(j).facecoordinates)<=3.5
	        neigh(3)=neigh(4)+1;
	   elseif EDist(events(i).facecoordinates,events(j).facecoordinates)>3.5 & EDist(events(i).facecoordinates,events(j).facecoordinates)<=4.5
	        neigh(4)=neigh(4)+1;
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
     if events(i).distance>0 & events(i).distance<=1.5
        conf(1)=conf(1)+1;
     elseif events(i).distance>1.5 & events(i).distance<=2.5
        conf(2)=conf(2)+1;
     elseif events(i).distance>2.5 & events(i).distance<=3.5
        conf(3)=conf(3)+1;
     elseif events(i).distance>3.5 & events(i).distance<=4.5
        conf(4)=conf(4)+1;
     end
     i=i+1;
  end
  
  %now find the probability  
  pconf=[conf(1)/neigh(1) conf(2)/neigh(2) conf(3)/neigh(3) conf(4)/neigh(4)]

  % it looks like there are no neighbors between 0 and .5 distance, so lets make the array smaller
  %  pconf=pconf(2:9)
  
  % Now, lets plot that mofo  
  x=1.5:1:4.5;
  y=pconf;
  plot(x,y, 'LineWidth',3)  
  publishfig
  xlabel('Distance Bins')
  ylabel('P(Confusion)')
  title('Confusability as a Function of Distance')
  pconf=pconf;
    

