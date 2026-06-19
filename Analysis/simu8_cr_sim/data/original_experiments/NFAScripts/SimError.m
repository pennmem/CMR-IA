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
cd ../../NFA19/session_0
ev19=intParFiles('NFA19',20,n);
cd ../../NFA20/session_0
ev20=intParFiles('NFA20',20,n);
cd ../../NFA21/session_0
ev21=intParFiles('NFA21',20,n);
cd ../../NFA22/session_0
ev22=intParFiles('NFA22',20,n);
cd ../../NFA23/session_0
ev23=intParFiles('NFA23',20,n);
cd ../../NFA24/session_0
ev24=intParFiles('NFA24',20,n);
cd ../../NFA25/session_0
ev25=intParFiles('NFA25',20,n);

%allevents=loadEvents('allevents.mat');

allevents=[ev1;ev2;ev3;ev4;ev5;ev6;ev7;ev8;ev9;ev10;ev11;ev12;ev13; ev14; ev15; ev16; ev17; ev18; ev19; ev20; ev21; ev22; ev23; ev24; ev25];
cd ../../
saveEvents(allevents,'allevents.mat')

pconf1=nfaSimilarity(ev1);
pconf2=nfaSimilarity(ev2);
pconf3=nfaSimilarity(ev3);
pconf4=nfaSimilarity(ev4);
pconf5=nfaSimilarity(ev5);
pconf6=nfaSimilarity(ev6);
pconf7=nfaSimilarity(ev7);
pconf8=nfaSimilarity(ev8);
pconf9=nfaSimilarity(ev9);
pconf10=nfaSimilarity(ev10);
pconf11=nfaSimilarity(ev11);
pconf12=nfaSimilarity(ev12);
pconf13=nfaSimilarity(ev13);
pconf14=nfaSimilarity(ev14);
pconf15=nfaSimilarity(ev15);
pconf16=nfaSimilarity(ev16);
pconf17=nfaSimilarity(ev17);
pconf18=nfaSimilarity(ev18);
pconf19=nfaSimilarity(ev19);
pconf20=nfaSimilarity(ev20);
pconf21=nfaSimilarity(ev21);
pconf22=nfaSimilarity(ev22);
pconf23=nfaSimilarity(ev23);
pconf24=nfaSimilarity(ev24);
pconf25=nfaSimilarity(ev25);

pconf=[pconf1; pconf2; pconf3; pconf4; pconf5; pconf6; pconf7; pconf8; pconf9; pconf10; pconf11; pconf12; pconf13; pconf14; pconf15; pconf16; pconf17; pconf18; pconf19; pconf20; pconf21; pconf22; pconf23; pconf24; pconf25]

x=[1:.5:4.5]

err1=pconf(:,1);
err1=[err1(2:13); err1(15:16); err1(20:22); err1(24)];
err2=pconf(:,2);
err2=[err2(1:19)];
err3=pconf(:,3);
err4=pconf(:,4);
err5=pconf(:,5);
err6=pconf(:,6);
err7=pconf(:,7);
err8=pconf(:,8);
err8=[0; 0; 0; 0; 0; 0; 0; .0500; 0; 0; 0; 0; .0500; 0];

m=[mean(err1) mean(err2) mean(err3) mean(err4) mean(err5) mean(err6) mean(err7) mean(err8)];

[H,P,C1]=ttest(err1,m(1),.05);
[H,P,C2]=ttest(err2,m(2),.05);
[H,P,C3]=ttest(err3,m(3),.05);
[H,P,C4]=ttest(err4,m(4),.05);
[H,P,C5]=ttest(err5,m(5),.05);
[H,P,C6]=ttest(err6,m(6),.05);
[H,P,C7]=ttest(err7,m(7),.05);
[H,P,C8]=ttest(err8,m(8),.05);

e = [C1(2)-C1(1) C2(2)-C2(1) C3(2)-C3(1) C4(2)-C4(1) C5(2)-C5(1) C6(2)-C6(1) C7(2)-C7(1) C8(2)-C8(1)]/2;

plot(x,m,'bo-','LineWidth',3)
hold on
errorbar(x,m,e,'b')
publishfig

%title('Confusability as a Function of Distance')
xlabel('Distance Bins')
ylabel('P(Intra-List Intrusion)')


