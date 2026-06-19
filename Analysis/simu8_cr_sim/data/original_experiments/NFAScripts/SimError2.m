% script that creates and makes error bars for a p(confusion) v. distance bins graph

%A "neighbor" is within n of the probe face
n=3;
cd NFA01/session_0
ev1=intParFilesB('NFA01',20,n);
cd ../../NFA02/session_0
ev2=intParFilesB('NFA02',20,n);
cd ../../NFA03/session_0
ev3=intParFilesB('NFA03',20,n);
cd ../../NFA04/session_0
ev4=intParFiles('NFA04',20,n);
cd ../../NFA05/session_0
ev5=intParFiles('NFA05',20,n);
cd ../../NFA06/session_0
ev6=intParFiles('NFA06',20,n);
cd ../../NFA07/session_0
ev7=intParFiles('NFA07',20,n);
cd ../../NFA08/session_0
ev8=intParFiles('NFA08',20,n);
cd ../../NFA09/session_0
ev9=intParFiles09('NFA09',20,n);
cd ../../NFA10/session_0
ev10=intParFiles('NFA10',20,n);
cd ../../NFA11/session_0
ev11=intParFiles('NFA11',20,n);
cd ../../NFA12/session_0
ev12=intParFiles12('NFA12',20,n);
cd ../../NFA13/session_0
ev13=intParFiles('NFA13',20,n);
cd ../../NFA14/session_0
ev14=intParFiles('NFA14',20,n);
cd ../../NFA15/session_0
ev15=intParFiles('NFA15',20,n);
cd ../../NFA16/session_0
ev16=intParFiles('NFA16',20,n);
cd ../../NFA17/session_0
ev17=intParFiles('NFA17',20,n);
cd ../../NFA18/session_0
ev18=intParFiles('NFA18',20,n);

%allevents=loadEvents('allevents.mat');

allevents=[ev1;ev2;ev3;ev4;ev5;ev6;ev7;ev8;ev9;ev10;ev11;ev12;ev13; ev14; ev15; ev16; ev17; ev18];
cd ../../
saveEvents(allevents,'allevents.mat')

pconf1=nfaSimilarity2(ev1);
pconf2=nfaSimilarity2(ev2);
pconf3=nfaSimilarity2(ev3);
pconf4=nfaSimilarity2(ev4);
pconf5=nfaSimilarity2(ev5);
pconf6=nfaSimilarity2(ev6);
pconf7=nfaSimilarity2(ev7);
pconf8=nfaSimilarity2(ev8);
pconf9=nfaSimilarity2(ev9);
pconf10=nfaSimilarity2(ev10);
pconf11=nfaSimilarity2(ev11);
pconf12=nfaSimilarity2(ev12);
pconf13=nfaSimilarity2(ev13);
pconf14=nfaSimilarity2(ev14);
pconf15=nfaSimilarity2(ev15);
pconf16=nfaSimilarity2(ev16);
pconf17=nfaSimilarity2(ev17);
pconf18=nfaSimilarity2(ev18);

pconf=[pconf1; pconf2; pconf3; pconf4; pconf5; pconf6; pconf7; pconf8; pconf9; pconf10; pconf11; pconf12; pconf13; pconf14; pconf15; pconf16; pconf17; pconf18]

x=[1.5:1:4.5]

err1=pconf(:,1);
err2=pconf(:,2);
err3=pconf(:,3);
err4=pconf(:,4);

m=[mean(err1) mean(err2) mean(err3) mean(err4)]

[H,P,C1]=ttest(err1,m(1),.05);
[H,P,C2]=ttest(err2,m(2),.05);
[H,P,C3]=ttest(err3,m(3),.05);
[H,P,C4]=ttest(err4,m(4),.05);

e = [C1(2)-C1(1) C2(2)-C2(1) C3(2)-C3(1) C4(2)-C4(1)]

plot(x,m,'b','LineWidth',3)
hold on
errorbar(x,m,e,'b')
publishfig

title('Confusability as a Function of Distance')
xlabel('Distance Bins')
ylabel('P(Confusion)')


