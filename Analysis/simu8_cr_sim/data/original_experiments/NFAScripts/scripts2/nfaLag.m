function pconf = nfaLag(events)

% FUNCTION pconf = nfaLag(events)
%  Finds all of the possible errors that could be made, calculates 
%  the temporal lags at study of those errors, then finds the 
%  probability based on the distances taht a confusion will be made
% 
% Input Arg:
%  events=events:structure you want to analyze
% Output Arg:
%  pconf=matrix of probabilities of making an error based on lag distance

  allevents=events;
  for e=1:length(allevents)
     allevents(e).lag=[];
  end
  
% add a field in the events structure for temporal lag at study, and fill it in
% i=the test item to be amended.
% j=the study item which has the face which corresponds to the recalled name
% k=the study item which has the face which corresponds to the test face 
  
  i=1;
  while i<=length(allevents)
     if allevents(i).typeevent==2 | allevents(i).typeevent==3     
        for j=1:length(allevents)
            if strcmp(allevents(i).recallname,allevents(j).facename)==1 & allevents(j).typeevent==1 & strcmp(allevents(i).subject,allevents(j).subject)==1 & allevents(i).blocknum==allevents(j).blocknum & allevents(i).vocalization==0 & allevents(i).intrusion==0
                for k=1:length(allevents)
                      if allevents(k).typeevent==1 & allevents(k).face==allevents(i).face & strcmp(allevents(j).subject,allevents(k).subject)==1 & allevents(j).blocknum==allevents(k).blocknum
	                   allevents(i).lag=(allevents(k).trialnum-allevents(j).trialnum);
                           allevents(k).lag=allevents(i).lag;
                      end
                end
            end
        end
     end
     i=i+1;
  end
  
  % Find all the possible lags
        
     
  % filter the structure so that it only includes 'study' items
  events=filterStruct(allevents, 'typeevent==1');

            
  % find all the possible neighbors and the confusions and 
  % from that find the probability of confusion based on temporal lag
  % lags=[-7 -6 -5 -4 -3 -2 -1 1 2 3 4 5 6 7]
  % conf=[-7 -6 -5 -4 -3 -2 -1 1 2 3 4 5 6 7]

  lag=[length(events)*(1/8) length(events)*(2/8) length(events)*(3/8) length(events)*(4/8) length(events)*(5/8) length(events)*(6/8) length(events)*(7/8) length(events)*(7/8) length(events)*(6/8) length(events)*(5/8) length(events)*(4/8) length(events)*(3/8) length(events)*(2/8) length(events)*(1/8)];
  conf=[0 0 0 0 0 0 0 0 0 0 0 0 0 0];
  
%  for i=1:length(events)
%     for j=1:length(events)
%        if strcmp(events(i).subject,events(j).subject)==1 & events(i).blocknum==events(j).blocknum & events(j).trialnum==1
%           for k=j:j+7
%	      if events(i).trialnum-events(k).trialnum==-7
%	         lag(1)=lag(1)+1;
%	      elseif events(i).trialnum-events(k).trialnum==-6
%                 lag(2)=lag(2)+1;
%              elseif events(i).trialnum-events(k).trialnum==-5
%	         lag(3)=lag(3)+1;
%	      elseif events(i).trialnum-events(k).trialnum==-4
%                 lag(4)=lag(4)+1;
%              elseif events(i).trialnum-events(k).trialnum==-3
%	         lag(5)=lag(5)+1;
%              elseif events(i).trialnum-events(k).trialnum==-2
%                 lag(6)=lag(6)+1;
%              elseif events(i).trialnum-events(k).trialnum==-1
%                 lag(7)=lag(7)+1;
%              elseif events(i).trialnum-events(k).trialnum==1
%                 lag(8)=lag(8)+1;
%              elseif events(i).trialnum-events(k).trialnum==2
%                 lag(9)=lag(9)+1;
%              elseif events(i).trialnum-events(k).trialnum==3
%                 lag(10)=lag(10)+1;
%              elseif events(i).trialnum-events(k).trialnum==4
%                 lag(11)=lag(11)+1;
%              elseif events(i).trialnum-events(k).trialnum==5
%                 lag(12)=lag(12)+1;
%              elseif events(i).trialnum-events(k).trialnum==6
%                lag(13)=lag(13)+1;
%              elseif events(i).trialnum-events(k).trialnum==7
%	         lag(14)=lag(14)+1;
%              end
%           end
%        end
%     end
%   end


%   Filter the structure so that only 'recall' events are shown and there are no confusions or vocalizations, then find the number of confusions for every lag bin
  events=filterStruct(allevents,'typeevent==3');
  events=filterStruct(events,'vocalization==0 & intrusion==0');
 
  
  for i=1:length(events)
     for j=-7:-1
         if events(i).lag==j
            conf(j+8)=conf(j+8)+1;
         end
     end 
     for j=1:7
         if events(i).lag==j
            conf(j+7)=conf(j+7)+1;
         end
     end
  end
  conf

  
%   now find the probability, and see which have non-zero values
  pconf=conf./lag;

%  Now, lets plot that mofo  
  x=[-7 -6 -5 -4 -3 -2 -1 1 2 3 4 5 6 7];
  y=pconf;
  plot(x,y,'r-', 'LineWidth',3)
  publishfig
  xlabel('Lag at Study')
  ylabel('P(Confusion)')
  title('Confusability as a Function of Lag at Study')  
    
  pconf = pconf;
