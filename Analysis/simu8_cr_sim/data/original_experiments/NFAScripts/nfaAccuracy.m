function [acc,rt] = nfaAccuracy(events,accuracy,reactiontime)

% FUNCTION [acc,acc1,acc2,rt, rt1, rt2] = nfaAccuracy(events,accuracy,rt)
% Calculate and plot accuracy and rt for data from an events structure
%
% Input Arg:
%  events='allEvents3_5':structure you want to analyze
%  accuracy= 0=no, 1=yes:do you want to analyze and plot accuracy
%  rt= 0=no, 1=yes: do you want to analyze and plot reaction time
% Output Arg
%  acc,acc1, acc2 = arrays of accuracies for combined, first half, second half
%  rt, rt1, rt2 = arrays of accuracies for combined, first half, second half

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

  acc=[];
  for j=1:10
     blockstruct=[];
     for k=1:length(accstruct)
        if accstruct(k).blocknum==j | accstruct(k).blocknum==(j+10)
            blockstruct=[blockstruct; accstruct(k)];
        end
     end
     if isempty(blockstruct)
        acc(j)=(0/0);
     else 
        iscorrect=[blockstruct.iscorrect];
        acc(j)=mean(iscorrect);
     end   
  end
	
      

  

% now, lets do some reaction time analysis.  This includes all recalls, regardless of whether or not its correct
% filter the structure so that it only includes 'recall' items and no vocalizations
            
  
  accstruct=filterStruct(events,'typeevent==3 & vocalization~=1');
    
  rt=[];
  for j=1:10
     blockstruct=[];
     for k=1:length(accstruct)     
        if accstruct(k).blocknum==j | accstruct(k).blocknum==j+10
           blockstruct=[blockstruct; accstruct(k)];
        end
     end
     if isempty(blockstruct)
        rt(j)=(0/0);
     else
        reactiontime=[blockstruct.rt];
        rt(j)=mean(reactiontime);
     end
  end
 
  
     

    

  

  % now, plot the reaction time bar charts
  % plot the accuracy bar charts
  
%  x=1:1:10;
%  if accuracy==1
%     subplot(2,3,1);
%     bar(x,y1p)
%     subplot(2,3,2);
%     bar(x,y2p)
%     subplot(2,3,3);
%     bar(x,yp)
    
%  end
%  if reactiontime==1
%     subplot(2,3,4);
%     plot(x,y1rt,'kx-','LineWidth',3)
%     subplot(2,3,5);
%     plot(x,y2rt,'kx-','LineWidth',3)
%     subplot(2,3,6);
%     plot(x,yrt,'kx-','LineWidth',3)
%  end
%  
%  publishfig
%  hold on
%
%  if accuracy==1
%     subplot(2,3,1);
%     xlabel('Block Number')
%     ylabel('P(Correct Recall)')
%     title('Accuracy by Block, First Half')
%     subplot(2,3,2);
%     title('Second Half')
%     subplot(2,3,3);
%     title('Combined')
%  end

%  hold on 

%  if reactiontime==1
%     subplot(2,3,4);
%     xlabel('Block Number')
%     ylabel('Average Reaction Time (ms)')
%     title('Reaction Time by Block, First Half')
%     subplot(2,3,5);
%     title('Second Half')
%     subplot(2,3,6);
%     title('Combined')
%  end

% Output arguments:

  acc=acc;
 % acc1=y1p;
 % acc2=y2p;
  rt=rt; 
 % rt1=y1rt;
 % rt2=y2rt;
