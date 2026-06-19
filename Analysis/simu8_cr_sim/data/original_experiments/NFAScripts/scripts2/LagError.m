% script that creates and makes error bars for a p(confusion) v. lag at study graph

%These next lines only need to be used if SimError is not run first
%allevents=loadEvents('allevents.mat');

%n=3;
%cd NFA01/session_0
%ev1=intParFilesB('NFA01',20,n);
%cd ../../NFA02/session_0
%ev2=intParFilesB('NFA02',20,n);
%cd ../../NFA03/session_0
%ev3=intParFilesB('NFA03',20,n);
%cd ../../NFA04/session_0
%ev4=intParFiles('NFA04',20,n);
%cd ../../NFA05/session_0
%ev5=intParFiles('NFA05',20,n);

pconf1=nfaLag(ev1);
pconf2=nfaLag(ev2);
pconf3=nfaLag(ev3);
pconf4=nfaLag(ev4);
pconf5=nfaLag(ev5);
pconf6=nfaLag(ev6);
pconf7=nfaLag(ev7);
pconf8=nfaLag(ev8);
pconf9=nfaLag(ev9);
pconf10=nfaLag(ev10);
pconf11=nfaLag(ev11);
pconf12=nfaLag(ev12);
pconf13=nfaLag(ev13);
pconf14=nfaLag(ev14);
pconf15=nfaLag(ev15);
pconf16=nfaLag(ev16);
pconf17=nfaLag(ev17);
pconf18=nfaLag(ev18);
pconf19=nfaLag(ev19);
pconf20=nfaLag(ev20);

pconf=[pconf1; pconf2; pconf3; pconf4; pconf5; pconf6; pconf7; pconf8; pconf9; pconf10; pconf11; pconf12; pconf13; pconf14; pconf15; pconf16; pconf17; pconf18; pconf19; pconf20]

x=[-7 -6 -5 -4 -3 -2 -1 1 2 3 4 5 6 7];
m=mean(pconf,1);

err1=pconf(:,1);
err2=pconf(:,2);
err3=pconf(:,3);
err4=pconf(:,4);
err5=pconf(:,5);
err6=pconf(:,6);
err7=pconf(:,7);
err8=pconf(:,8);
err9=pconf(:,9);
err10=pconf(:,10);
err11=pconf(:,11);
err12=pconf(:,12);
err13=pconf(:,13);
err14=pconf(:,14);

[H,P,C1]=ttest(err1,m(1),.05);
[H,P,C2]=ttest(err2,m(2),.05);
[H,P,C3]=ttest(err3,m(3),.05);
[H,P,C4]=ttest(err4,m(4),.05);
[H,P,C5]=ttest(err5,m(5),.05);
[H,P,C6]=ttest(err6,m(6),.05);
[H,P,C7]=ttest(err7,m(7),.05);
[H,P,C8]=ttest(err8,m(8),.05);
[H,P,C9]=ttest(err9,m(9),.05);
[H,P,C10]=ttest(err10,m(10),.05);
[H,P,C11]=ttest(err11,m(11),.05);
[H,P,C12]=ttest(err12,m(12),.05);
[H,P,C13]=ttest(err13,m(13),.05);
[H,P,C14]=ttest(err14,m(14),.05);

e = [C1(2)-C1(1) C2(2)-C2(1) C3(2)-C3(1) C4(2)-C4(1) C5(2)-C5(1) C6(2)-C6(1) C7(2)-C7(1) C8(2)-C8(1) C9(2)-C9(1) C10(2)-C10(1) C11(2)-C11(1) C12(2)-C12(1) C13(2)-C13(1) C14(2)-C14(1)]/2;

plot(x,m,'r','LineWidth',3)
hold on
errorbar(x,m,e, 'r')
publishfig

title('Confusability as a Function of Lag at Study')
xlabel('Lag at Study')
ylabel('P(Confusion)')
