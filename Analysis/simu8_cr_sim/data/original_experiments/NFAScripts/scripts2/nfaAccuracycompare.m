function nfaAccuracy(events,accuracy,reactiontime)

% FUNCTION nfaAccuracy(events,accuracy,rt)
% Calculate and plot accuracy and rt for data from an events structure
%
% Input Arg:
%  events='allEvents3_5':structure you want to analyze
%  accuracy= 0=no, 1=yes:do you want to analyze and plot accuracy
%  rt= 0=no, 1=yes: do you want to analyze and plot reaction time

%  if ~exist(events)
%     error('You must enter an existing events structure')
%  end
  if isempty(accuracy)
     error('You must enter 0 or 1 for accuracy field')
  end
  if isempty(reactiontime)
     error('You must enter 0 or 1 for accuracy field')
  end

% first load the structure
%  events=loadEvents(events);


% now, get accuracy for half1, half2, and combined
% filter the structure so it only includes test items
  accstruct=filterStruct(events,'typeevent==2');

  y1=[0 0 0 0 0 0 0 0 0 0];
  y1total=[0 0 0 0 0 0 0 0 0 0]; 
  y1p=[0 0 0 0 0 0 0 0 0 0];
  for i=1:length(accstruct)
     for j=1:length(y1)     
        if accstruct(i).blocknum==j
          iscorrect=accstruct(i).iscorrect;
          y1(j)=y1(j)+iscorrect;
          y1total(j)=y1total(j)+1;
          y1p(j)=(y1(j)/y1total(j));
        end
     end    
  end

  y2=[0 0 0 0 0 0 0 0 0 0];
  y2total=[0 0 0 0 0 0 0 0 0 0]; 
  y2p=[0 0 0 0 0 0 0 0 0 0];
  for i=1:length(accstruct)
     for j=1:length(y2)     
        if accstruct(i).blocknum==j+10
          iscorrect=accstruct(i).iscorrect;
          y2(j)=y2(j)+iscorrect;
          y2total(j)=y2total(j)+1;
          y2p(j)=(y2(j)/y2total(j));
        end
     end    
  end 
  
  y=[0 0 0 0 0 0 0 0 0 0];
  ytotal=[0 0 0 0 0 0 0 0 0 0]; 
  yp=[0 0 0 0 0 0 0 0 0 0];
  for i=1:length(accstruct)
     for j=1:length(y)     
        if accstruct(i).blocknum==j | accstruct(i).blocknum==j+10
          iscorrect=accstruct(i).iscorrect;
          y(j)=y(j)+iscorrect;
          ytotal(j)=ytotal(j)+1;
          yp(j)=(y(j)/ytotal(j));
        end
     end    
  end
  

% now, lets do some reaction time analysis.  This includes all recalls, regardless of whether or not its correct
% filter the structure so that it only includes 'recall' items and no vocalizations
            
  
  accstruct=filterStruct(events,'typeevent==3 & vocalization~=1');
  
   
  y1=[0 0 0 0 0 0 0 0 0 0];
  y1total=[0 0 0 0 0 0 0 0 0 0]; 
  y1rt=[0 0 0 0 0 0 0 0 0 0];
  for i=1:length(accstruct)
     for j=1:length(y1)     
        if accstruct(i).blocknum==j
          rt=accstruct(i).rt;
          y1(j)=y1(j)+rt;
          y1total(j)=y1total(j)+1;
          y1rt(j)=(y1(j)/y1total(j));
        end
     end    
  end

  y2=[0 0 0 0 0 0 0 0 0 0];
  y2total=[0 0 0 0 0 0 0 0 0 0]; 
  y2rt=[0 0 0 0 0 0 0 0 0 0];
  for i=1:length(accstruct)
     for j=1:length(y2)     
        if accstruct(i).blocknum==j+10
          rt=accstruct(i).rt;
          y2(j)=y2(j)+rt;
          y2total(j)=y2total(j)+1;
          y2rt(j)=(y2(j)/y2total(j));
        end
     end    
  end 
  
  y=[0 0 0 0 0 0 0 0 0 0];
  ytotal=[0 0 0 0 0 0 0 0 0 0]; 
  yrt=[0 0 0 0 0 0 0 0 0 0];
  for i=1:length(accstruct)
     for j=1:length(y)     
        if accstruct(i).blocknum==j | accstruct(i).blocknum==j+10
          rt=accstruct(i).rt;
          y(j)=y(j)+rt;
          ytotal(j)=ytotal(j)+1;
          yrt(j)=(y(j)/ytotal(j));
        end
     end    
  end

  

  % now, plot the reaction time bar charts
  % plot the accuracy bar charts
  
  x=1:1:10;
  if accuracy==1
     subplot(2,3,1);
     plot(x,y1p,'bx-','LineWidth',3)
     hold on
     subplot(2,3,2);
     plot(x,y2p,'bx-','LineWidth',3)
     hold on
     subplot(2,3,3);
     plot(x,yp,'bx-','LineWidth',3)
     hold on
  end
  if reactiontime==1
     subplot(2,3,4);
     plot(x,y1rt,'bx-','LineWidth',3)
     hold on
     subplot(2,3,5);
     hold on
     plot(x,y2rt,'bx-','LineWidth',3)
     subplot(2,3,6);
     hold on
     plot(x,yrt,'bx-','LineWidth',3)
  end
  
  publishfig
  hold on

  if accuracy==1
     subplot(2,3,1);
     xlabel('Block Number')
     ylabel('P(Correct Recall)')
     title('Accuracy by Block, First Half')
     subplot(2,3,2);
     title('Second Half')
     subplot(2,3,3);
     title('Combined')
  end

  hold on 

  if reactiontime==1
     subplot(2,3,4);
     xlabel('Block Number')
     ylabel('Average Reaction Time (ms)')
     title('Reaction Time by Block, First Half')
     subplot(2,3,5);
     title('Second Half')
     subplot(2,3,6);
     title('Combined')
  end

