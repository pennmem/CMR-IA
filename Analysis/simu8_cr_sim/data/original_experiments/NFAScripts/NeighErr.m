%   NeighErr
%   Average across the subjects and plot Neighbors effect with error bars


acc1=nfaNeighbors(ev1);
acc2=nfaNeighbors(ev2);
acc3=nfaNeighbors(ev3);
acc4=nfaNeighbors(ev4);
acc5=nfaNeighbors(ev5);
acc6=nfaNeighbors(ev6);
acc7=nfaNeighbors(ev7);
acc8=nfaNeighbors(ev8);
acc9=nfaNeighbors(ev9);
acc10=nfaNeighbors(ev10);
acc11=nfaNeighbors(ev11);
acc12=nfaNeighbors(ev12);
acc13=nfaNeighbors(ev13);
acc14=nfaNeighbors(ev14);
acc15=nfaNeighbors(ev15);
acc16=nfaNeighbors(ev16);
acc17=nfaNeighbors(ev17);
acc18=nfaNeighbors(ev18);
acc19=nfaNeighbors(ev19);
acc20=nfaNeighbors(ev20);
acc21=nfaNeighbors(ev21);
acc22=nfaNeighbors(ev22);
acc23=nfaNeighbors(ev23);
acc24=nfaNeighbors(ev24);
acc25=nfaNeighbors(ev25);

acc1=nanmean(acc1,2);
acc2=nanmean(acc2,2);
acc3=nanmean(acc3,2);
acc4=nanmean(acc4,2);
acc5=nanmean(acc5,2);
acc6=nanmean(acc6,2);
acc7=nanmean(acc7,2);
acc8=nanmean(acc8,2);
acc9=nanmean(acc9,2);
acc10=nanmean(acc10,2);
acc11=nanmean(acc11,2);
acc12=nanmean(acc12,2);
acc13=nanmean(acc13,2);
acc14=nanmean(acc14,2);
acc15=nanmean(acc15,2);
acc16=nanmean(acc16,2);
acc17=nanmean(acc17,2);
acc18=nanmean(acc18,2);
acc19=nanmean(acc19,2);
acc20=nanmean(acc20,2);
acc21=nanmean(acc21,2);
acc22=nanmean(acc22,2);
acc23=nanmean(acc23,2);
acc24=nanmean(acc24,2);
acc25=nanmean(acc25,2);

allacc=[acc1 acc2 acc3 acc4 acc5 acc6 acc7 acc8 acc9 acc10 acc11 acc12 acc13 acc14 acc15 acc16 acc17 acc18 acc19 acc20 acc21 acc22 acc23 acc24 acc25]

m=nanmean(allacc,2)


