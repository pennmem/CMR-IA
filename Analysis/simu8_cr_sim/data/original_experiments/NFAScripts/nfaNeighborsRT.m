function rt = nfaNeighborsRT(events)

%  Function rt = nfaNeighborsRT(events)
%  Takes an events structure from nfa and determines the reaction time for
%  faces with each amount of neighbors in similarity space (1,2,3,4,5,6,7).
%     Inputs:
%         1. events = events, events structure
%     Outputs:
%         1. rt =  reaction time matrix for each of the 
%                  seven neighbor possibilities,
%                  for now theres not enough data 
%                  for 7 so Im leaving it out
%                  theres also optional code to combine
%                  them with 6, valid? With more data,
%                  we can use 7

  %Filter the structure for only test events

  events=filterStruct(events,'typeevent==2');

 
  
  for b=1:10
     eventsb=filterStruct(events,sprintf('blocknum==%d',b));
     eventsbb=filterStruct(events,sprintf('blocknum==%d',(b+10)));
     eventsb=[eventsb;eventsbb];
     j=1;
     while j <=7
        eventsj=filterStruct(eventsb, sprintf('neighbors==%d', j));
        accj=[eventsj(1:length(eventsj)).rt];
        accmatrix(j)=mean(accj);
        j=j+1;
     end
    % eventsj=filterStruct(eventsb, sprintf('neighbors>4'))
    % accj=[eventsj(1:length(eventsj)).rt];
    % accmatrix(5)=mean(accj);
      acc(:,b)=accmatrix; 
  end
  
  x=[1:7];
  for b=1:10
      l=corrcoef(x,acc(:,b));
      r(b)=l(2); 
  end
  
  x=[1:10];
  plot(x,r,'r','LineWidth',3)
  hold on
  y=[0 0 0 0 0 0 0 0 0 0]
  plot(x,y,'k--')
  publishfig

  title('Correlation between # of Neighbors and Reaction Time, by Block (A "neighbor" is within 3 units of the probe face)')
  ylabel('Correlation Coefficient (r)')
  xlabel('Block Number')
  rt=acc;

