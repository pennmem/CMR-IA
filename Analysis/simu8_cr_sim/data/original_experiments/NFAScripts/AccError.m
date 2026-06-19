%plot nfa Accuracy and RT with error bars
%first, must run nfaPrep

acc=[];
rt=[];

for i=1:25
  mystr=['events=ev' int2str(i) ';'];
  eval(mystr);
  [subj_acc, subj_rt]=nfaAccuracy(events,1,1);
  acc=[acc; subj_acc];
  rt=[rt; subj_rt];
end



macc=nanmean(acc,1)

mrt=nanmean(rt,1)


err1acc=acc(:,1);
err2acc=acc(:,2);
err3acc=acc(:,3);
err4acc=acc(:,4);
err5acc=acc(:,5);
err6acc=acc(:,6);
err7acc=acc(:,7);
err8acc=acc(:,8);
err9acc=acc(:,9);
err10acc=acc(:,10);


[H,P,C1]=ttest(err1acc,macc(1),.05);
[H,P,C2]=ttest(err2acc,macc(2),.05);
[H,P,C3]=ttest(err3acc,macc(3),.05);
[H,P,C4]=ttest(err4acc,macc(4),.05);
[H,P,C5]=ttest(err5acc,macc(5),.05);
[H,P,C6]=ttest(err6acc,macc(6),.05);
[H,P,C7]=ttest(err7acc,macc(7),.05);
[H,P,C8]=ttest(err8acc,macc(8),.05);
[H,P,C9]=ttest(err9acc,macc(9),.05);
[H,P,C10]=ttest(err10acc,macc(10),.05);


err1rt=rt(:,1);
err2rt=rt(:,2);
err3rt=rt(:,3);
err4rt=rt(:,4);
err5rt=rt(:,5);
err6rt=rt(:,6);
err7rt=rt(:,7);
err8rt=rt(:,8);
err9rt=rt(:,9);
err10rt=rt(:,10);

[H,P,R1]=ttest(err1rt,mrt(1),.05);
[H,P,R2]=ttest(err2rt,mrt(2),.05);
[H,P,R3]=ttest(err3rt,mrt(3),.05);
[H,P,R4]=ttest(err4rt,mrt(4),.05);
[H,P,R5]=ttest(err5rt,mrt(5),.05);
[H,P,R6]=ttest(err6rt,mrt(6),.05);
[H,P,R7]=ttest(err7rt,mrt(7),.05);
[H,P,R8]=ttest(err8rt,mrt(8),.05);
[H,P,R9]=ttest(err9rt,mrt(9),.05);
[H,P,R10]=ttest(err10rt,mrt(10),.05);

e = [C1(2)-C1(1) C2(2)-C2(1) C3(2)-C3(1) C4(2)-C4(1) C5(2)-C5(1) C6(2)-C6(1) C7(2)-C7(1) C8(2)-C8(1) C9(2)-C9(1) C10(2)-C10(1)]/2;

r = [R1(2)-R1(1) R2(2)-R2(1) R3(2)-R3(1) R4(2)-R4(1) R5(2)-R5(1) R6(2)-R6(1) R7(2)-R7(1) R8(2)-R8(1) R9(2)-R9(1) R10(2)-R10(1)]/2;


x=1:1:10;
     subplot(1,2,1);
     plot(x,macc,'ko--','LineWidth',3)
     errorbar(x,macc,e,'k')
     hold on
     subplot(1,2,2);
     plot(x,mrt,'k^--')
     errorbar(x,mrt,r,'k')
     
 publishfig
 hold on

 
     subplot(1,2,1);
     xlabel('Block Number')
     ylabel('Probability of Recall)')
   
 hold on

 
     subplot(1,2,2);
     xlabel('Block Number')
     ylabel('Reaction Time (ms)')
     
