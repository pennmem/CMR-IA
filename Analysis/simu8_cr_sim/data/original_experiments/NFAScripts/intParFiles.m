function allEvents = intParFiles(subject,numBlocks,neighpara)

%script intParFiles
%
%
% Integrate parse files into the rest of the event structure, 
% by first constructing new response events and then adding new 
% fields to test events with the new information obtained. 
% ONLY WORKS FOR EXP W/8 FACES PER STUDY/TEST BLOCK, 
% 20 TOTAL BLOCKS, AND FOR PRESENTATION TIMES OF 5000 ms 
% and INTERPRESENTATION TIMES OF 1000 ms: 
% if changed may have to be reconfigured.
%
%  
% INPUT ARGS:
%subject = 'NFA01';
%numBlocks = 20; number of total blocks i.e. par files included in the experiment
%neighpara = 3; neighbor parameter: How many units away from the face do you want to be considered a "neighbor"?
%
% OUTPUT ARGS:
%   allEvents- An events structure that you then can save to 
%           /allevents.mat and can load when needed.  This event struct is
%           a new structure with parse data integrated into it. Has
%           the following fields: in future revisions, also the number of 
%           neighbors and coordinates and the distance of each face 
%           may be added.
%   1)subject = subject number
%   2)mstime = time of event
%   3)msoffset = inaccuracy of runtime
%   4)trialnum = trial number in block
%   5)typeevent = type of event
%   6)halfnum = which half of the session does this event occur in?
%   7)blocknum = which block of the session does this event occur in?
%   8)face = which face is being shown on the screen (a number).
%   9)facename = what would be the correctly associated name to that face in this session
%   10)msresponse = if it is a test event,and there is a reponse, this is the time when the person responded
%   11)rt = reaction time for the response, should also be included for study trials
%   12)recallname = the name recalled at test
%   13)recallface = the correctly associated face (number) to the name recalled, whether it be a correct response or a confusion or a prior-list intrusion
%   14)iscorrect = is recallname equal to facename? 0=no, 1=yes, should also be included for study trials
%   15)intrusion = intrusion? 0=no 1=yes.  Be careful, intrusions can be on in the namepool if they have not yet been seen by the subject

  if isempty(subject)
     error('You must enter subject number')
  end

  if isempty(numBlocks)
     error('You must enter number of blocks i.e. par files')
  end
  
  if isempty(neighpara)
     error('You must enter in a neighbor parameter')
  end
 
% create the original events structure from session.log with createEvents
  events = createEvents('session.log','%n%n%n%s%n%n%n%s',{'mstime','msoffset','trialnum','typeevent','halfnum','blocknum','face','facename'});
 
% add subject field and fill it in, create new fields in the structure 
  for i = 1:length(events)
     events(i).subject=subject;
     events(i).facecoordinates=[];
     events(i).neighbors=[];
     events(i).msresponse=[];
     events(i).rt=[];
     events(i).recallname=[];
     events(i).recallface=[];
     events(i).recallfacecoordinates=[];
     events(i).iscorrect=[];
     events(i).intrusion=[];
     events(i).vocalization=[];
  end
  
% load in PARFILES, create an event structure for every parfile loaded
  par = 1;
  parevents = [];
  
  while par<=numBlocks    
% choose andload the PARFILE
    parfile = sprintf('%d.par',par);
    [recalltime,namenum,recallname] = textread(parfile,'%n%n%s');

% make events out of the lines in the PARFILE, add some information, and append them to the parevents structure
    e=1;
    while e<=length(recalltime)
       event=struct('subject',[subject],'mstime',[events(9+((par-1)*17)).mstime+recalltime(e)],'msoffset',[0],'trialnum',[],'typeevent',['recall'],'halfnum',[],'blocknum',[par],'face',[],'facename',[],'facecoordinates',[],'neighbors',[],'msresponse',[events(9+((par-1)*17)).mstime+recalltime(e)],'rt',[],'recallname',[recallname(e)],'recallface',[],'recallfacecoordinates',[],'iscorrect',[],'intrusion',[],'vocalization',[]);
       if par<=(numBlocks/2)
         event.halfnum=1;
       else
         event.halfnum=2;
       end
       if namenum(e)==-1 & strcmp(recallname(e),'VV')==0
         event.intrusion=1;
       else
         event.intrusion=0;
       end
   
       parevents=[parevents; event];
        e=e+1;   
    end

    par = par+1;          
  end
  
  % fill in the some of the new fields in the old structure and the empty fields in the new parfile structure
  e=1;
  while e<=length(parevents)
    for test=1:length(events)
      if strcmp(events(test).typeevent,'test')==1
        if (parevents(e).mstime-events(test).mstime)<6500 & (parevents(e).mstime-events(test).mstime)>400
          parevents(e).face = events(test).face; 
          parevents(e).facename = events(test).facename;
          parevents(e).rt = parevents(e).mstime-events(test).mstime;
          parevents(e).iscorrect = strcmp(parevents(e).recallname,events(test).facename);
          events(test).msresponse = parevents(e).mstime;
          events(test).rt = parevents(e).mstime-events(test).mstime;
          events(test).iscorrect=strcmp(events(test).facename,parevents(e).recallname);
          events(test).recallname = parevents(e).recallname;
          events(test).intrusion=parevents(e).intrusion;
        end
      end
    end
  % see what face matches the name recalled by the subject at test(may or may not be correct)
    for recallnameindex=1:8
        if strcmp(parevents(e).recallname,events(recallnameindex).facename)==1 & parevents(e).intrusion~=1
          parevents(e).recallface = events(recallnameindex).face;
	end
    end
    for recallnameindex=171:178
        if strcmp(parevents(e).recallname,events(recallnameindex).facename)==1 & parevents(e).intrusion~=1
          parevents(e).recallface = events(recallnameindex).face;
        end
    end
    e=e+1;
  end 

 % fill in empty fields in test events with no corresponding recall event
  for test=1:length(events)
    if isempty(events(test).iscorrect) & strcmp(events(test).typeevent,'test')==1
       events(test).iscorrect=0;
       events(test).intrusion=0;
    end
    for recallnameindex=1:8
       if strcmp(events(test).recallname, events(recallnameindex).facename)==1 & events(test).intrusion~=1
         events(test).recallface=events(recallnameindex).face;
       end
    end
    for recallnameindex=171:178
       if strcmp(events(test).recallname, events(recallnameindex).facename)==1 & events(test).intrusion~=1
         events(test).recallface=events(recallnameindex).face;
       end
    end
  end 

 % concatenate the new structure to include all study/test/record and recall(parse) events
  events = [events; parevents];

 % add in coordinate information for calculating distance between faces/neighbors
 % first, the facecoordinates
  for i=1:length(events)
     if events(i).face==0
       events(i).facecoordinates=[.901 .706 1.060 -1.079];
     elseif events(i).face==1
       events(i).facecoordinates=[1.886 -.683 -.227 .498];
     elseif events(i).face==2 
       events(i).facecoordinates=[1.128 .942 .022 -1.137];
     elseif events(i).face==3
       events(i).facecoordinates=[1.284 -1.257 -.231 .891];
     elseif events(i).face==4
       events(i).facecoordinates=[-.075 1.511 -1.029 -.483];
     elseif events(i).face==5
       events(i).facecoordinates=[.484 -.390 -1.059 1.690]; 
     elseif events(i).face==6
       events(i).facecoordinates=[.384 1.504 -.951 .552];
     elseif events(i).face==7
       events(i).facecoordinates=[.268 -.468 -.106 2.199];
     elseif events(i).face==8
       events(i).facecoordinates=[-.723 .215 1.953 .039];
     elseif events(i).face==9
       events(i).facecoordinates=[-.248 -1.602 .534 -.860];
     elseif events(i).face==10
       events(i).facecoordinates=[-.342 .087 1.801 -.789];
     elseif events(i).face==11
       events(i).facecoordinates=[.402 -1.269 -.481 -1.391];
     elseif events(i).face==12
       events(i).facecoordinates=[-1.298 1.292 .925 .165];
     elseif events(i).face==13
       events(i).facecoordinates=[-1.249 -.599 -1.242 -.532];
     elseif events(i).face==14
       events(i).facecoordinates=[-1.725 .840 .301 -.444];
     elseif events(i).face==15
       events(i).facecoordinates=[-1.079 -.828 -1.271 .682];
     end
  % now, the recallfacecoordinates    
     if events(i).recallface==0
       events(i).recallfacecoordinates=[.901 .706 1.060 -1.079];
     elseif events(i).recallface==1
       events(i).recallfacecoordinates=[1.886 -.683 -.227 .498];
     elseif events(i).recallface==2 
       events(i).recallfacecoordinates=[1.128 .942 .022 -1.137];
     elseif events(i).recallface==3
       events(i).recallfacecoordinates=[1.284 -1.257 -.231 .891];
     elseif events(i).recallface==4
       events(i).recallfacecoordinates=[-.075 1.511 -1.029 -.483];
     elseif events(i).recallface==5
       events(i).recallfacecoordinates=[.484 -.390 -1.059 1.690]; 
     elseif events(i).recallface==6
       events(i).recallfacecoordinates=[.384 1.504 -.951 .552];
     elseif events(i).recallface==7
       events(i).recallfacecoordinates=[.268 -.468 -.106 2.199];
     elseif events(i).recallface==8
       events(i).recallfacecoordinates=[-.723 .215 1.953 .039];
     elseif events(i).recallface==9
       events(i).recallfacecoordinates=[-.248 -1.602 .534 -.860];
     elseif events(i).recallface==10
       events(i).recallfacecoordinates=[-.342 .087 1.801 -.789];
     elseif events(i).recallface==11
       events(i).recallfacecoordinates=[.402 -1.269 -.481 -1.391];
     elseif events(i).recallface==12
       events(i).recallfacecoordinates=[-1.298 1.292 .925 .165];
     elseif events(i).recallface==13
       events(i).recallfacecoordinates=[-1.249 -.599 -1.242 -.532];
     elseif events(i).recallface==14
       events(i).recallfacecoordinates=[-1.725 .840 .301 -.444];
     elseif events(i).recallface==15
       events(i).recallfacecoordinates=[-1.079 -.828 -1.271 .682];
     end
  % How about some sweet theoretical coordinates?
     if events(i).face==0
       events(i).theorfacecoordinates=[-1 -1 -1 -1];
     elseif events(i).face==1
       events(i).theorfacecoordinates=[-1 -1 -1 1];
     elseif events(i).face==2 
       events(i).theorfacecoordinates=[-1 -1 1 -1];
     elseif events(i).face==3
       events(i).theorfacecoordinates=[-1 -1 1 1];
     elseif events(i).face==4
       events(i).theorfacecoordinates=[-1 1 -1 -1];
     elseif events(i).face==5
       events(i).theorfacecoordinates=[-1 1 -1 1]; 
     elseif events(i).face==6
       events(i).theorfacecoordinates=[-1 1 1 -1];
     elseif events(i).face==7
       events(i).theorfacecoordinates=[-1 1 1 1];
     elseif events(i).face==8
       events(i).theorfacecoordinates=[1 -1 -1 -1];
     elseif events(i).face==9
       events(i).theorfacecoordinates=[1 -1 -1 1];
     elseif events(i).face==10
       events(i).theorfacecoordinates=[1 -1 1 -1];
     elseif events(i).face==11
       events(i).theorfacecoordinates=[1 -1 1 1];
     elseif events(i).face==12
       events(i).theorfacecoordinates=[1 1 -1 -1];
     elseif events(i).face==13
       events(i).theorfacecoordinates=[1 1 -1 1];
     elseif events(i).face==14
       events(i).theorfacecoordinates=[1 1 1 -1];
     elseif events(i).face==15
       events(i).theorfacecoordinates=[1 1 1 1];
     end
   % How about some sweet theoretical coordinates of the face of the recalled name?
     if events(i).recallface==0
       events(i).theorrecallfacecoordinates=[-1 -1 -1 -1];
     elseif events(i).recallface==1
       events(i).theorrecallfacecoordinates=[-1 -1 -1 1];
     elseif events(i).recallface==2 
       events(i).theorrecallfacecoordinates=[-1 -1 1 -1];
     elseif events(i).recallface==3
       events(i).theorrecallfacecoordinates=[-1 -1 1 1];
     elseif events(i).recallface==4
       events(i).theorrecallfacecoordinates=[-1 1 -1 -1];
     elseif events(i).recallface==5
       events(i).theorrecallfacecoordinates=[-1 1 -1 1]; 
     elseif events(i).recallface==6
       events(i).theorrecallfacecoordinates=[-1 1 1 -1];
     elseif events(i).recallface==7
       events(i).theorrecallfacecoordinates=[-1 1 1 1];
     elseif events(i).recallface==8
       events(i).theorrecallfacecoordinates=[1 -1 -1 -1];
     elseif events(i).recallface==9
       events(i).theorrecallfacecoordinates=[1 -1 -1 1];
     elseif events(i).recallface==10
       events(i).theorrecallfacecoordinates=[1 -1 1 -1];
     elseif events(i).recallface==11
       events(i).theorrecallfacecoordinates=[1 -1 1 1];
     elseif events(i).recallface==12
       events(i).theorrecallfacecoordinates=[1 1 -1 -1];
     elseif events(i).recallface==13
       events(i).theorrecallfacecoordinates=[1 1 -1 1];
     elseif events(i).recallface==14
       events(i).theorrecallfacecoordinates=[1 1 1 -1];
     elseif events(i).recallface==15
       events(i).theorrecallfacecoordinates=[1 1 1 1];
     end
  end

 % add in 4 dim MDS distances and theoretical distances between face shown and face of name recalled
  for v=1:length(events)
     if ~isempty(events(v).recallfacecoordinates)
       d=EDist(events(v).facecoordinates,events(v).recallfacecoordinates);
       events(v).distance=d;
     end
     if ~isempty(events(v).theorrecallfacecoordinates)
       d=EDist(events(v).theorfacecoordinates,events(v).theorrecallfacecoordinates);
       events(v).theordistance=d;
     end   
  end
  
 % find how many neighbors the presented face has and insert that number into the structure
 %n=number of neighbors for each of the 16 faces in this particular session 
  n=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
  for e=1:8
     for f=1:8
       dist = EDist(events(e).facecoordinates,events(f).facecoordinates);
       if e~=f & dist<neighpara
         n(e)=n(e)+1;
       end
     end
  end
  for e=171:178
     for f=171:178
       dist = EDist(events(e).facecoordinates,events(f).facecoordinates);
       if e~=f & dist<neighpara
         n(e-162)=n(e-162)+1;
       end
     end
  end 
  for event=1:length(events)
     for e=1:8
       if events(event).face==events(e).face
         events(event).neighbors=n(e);
       end
     end
     for e=171:178
       if events(event).face==events(e).face
         events(event).neighbors=n(e-162);
       end
     end
  end

  % change study/test/recall/record to a numerical value
  for event=1:length(events)
     if strcmp(events(event).typeevent,'study')==1
        events(event).typeevent=1; 
     elseif strcmp(events(event).typeevent,'test')==1
        events(event).typeevent=2;
     elseif strcmp(events(event).typeevent,'recall')==1
        events(event).typeevent=3;
     elseif strcmp(events(event).typeevent,'record')==1
        events(event).typeevent=4;
     end
     if strcmp(events(event).recallname,'VV')==1
        events(event).vocalization=1;
     else
        events(event).vocalization=0;
     end
  end

 % finally, generate outputs  
  allEvents = events;


       
