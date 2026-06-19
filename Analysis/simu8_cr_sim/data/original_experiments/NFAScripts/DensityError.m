%script DensityError

%i=1;
%densities=[];
%
%while i<=25
%    mystr = ['events=ev' int2str(i) ';'];
%    eval(mystr);
%    [one,two]=nfaDensity(events);
%    densities=[densities;one;two];
%    i=i+1;
%end
%mean=mean(densities)

mean=1.9261;

i=1;
while i<=25
   mystr = ['events=ev' int2str(i) ';'];
   eval(mystr);
   [one,two]=nfaDensity(events);
   half1=filterStruct(events,'halfnum==1');
   half2=filterStruct(events,'halfnum==2');
   if one<=mean & two>=mean
       string = ['ev' int2str(i) 'low=half1;'];
       eval(string);
       string = ['ev' int2str(i) 'high=half2;'];
       eval(string);
       status = ['Subject ' int2str(i) ' has one low and one high.']
   elseif one>=mean & two<=mean
       string = ['ev' int2str(i) 'high=half1;'];
       eval(string);
       string = ['ev' int2str(i) 'low=half2;'];
       eval(string);
       status = ['Subject ' int2str(i) ' has one low and one high.']
   elseif one<=mean & two<=mean
       string = ['ev' int2str(i) 'low=events;'];
       eval(string);
       string = ['ev' int2str(i) 'high=[];'];
       eval(string);
       status = ['Subject ' int2str(i) ' has two low density halves.']
   elseif one>=mean & two>=mean
       string = ['ev' int2str(i) 'high=events;'];
       eval(string);
       string = ['ev' int2str(i) 'low=[];'];
       eval(string);
       status = ['Subject ' int2str(i) ' has two high density halves.']
   end
   i=i+1;
end   

%Calculate neighborhood effect for low density lists

i=1;
lowneigh=[];
lowslope=[];
lowrt=[];
lowrtslope=[];
while i<=25
    mystr = ['events=ev' int2str(i) 'low;'];
    eval(mystr);
    if ~isempty(events)
%         [stuff]=nfaNeighborhood(events)
%         lowneigh=[lowneigh; neigh];
%         lowslope=[lowslope; slope];
%         lowrt=[lowrt; rt];
%         lowrtslope=[lowrtslope; rtslope];
          i=i+1;
    else
          i=i+1;
    end      
end

%Calculate neighborhood effect for high density lists

i=1;
highneigh=[];
highslope=[];
highrt=[];
highrtslope=[];
while i<=25;
    mystr = ['events=ev' int2str(i) 'high;'];
    eval(mystr);
    if ~isempty(events)
%         [stuff]=nfaNeighborhood(events)
%         highneigh=[lowneigh; neigh];
%         highslope=[highslope; slope];
%         highrt=[highrt; rt];
%         highrtslope=[highrtslope; rtslope];
          i=i+1;
    else
          i=i+1;
    end
end

%Compare the conditions

