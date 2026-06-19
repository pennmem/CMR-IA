% script that creates and makes error bars for a p(confusion) v. theoretical distance graph

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

pconf1=nfaTheorSim(ev1);
pconf2=nfaTheorSim(ev2);
pconf3=nfaTheorSim(ev3);
pconf4=nfaTheorSim(ev4);
pconf5=nfaTheorSim(ev5);
pconf6=nfaTheorSim(ev6);
pconf7=nfaTheorSim(ev7);
pconf8=nfaTheorSim(ev8);
pconf9=nfaTheorSim(ev9);
pconf10=nfaTheorSim(ev10);
pconf11=nfaTheorSim(ev11);
pconf12=nfaTheorSim(ev12);
pconf13=nfaTheorSim(ev13);
pconf14=nfaTheorSim(ev14);
pconf15=nfaTheorSim(ev15);
pconf16=nfaTheorSim(ev16);
pconf17=nfaTheorSim(ev17);
pconf18=nfaTheorSim(ev18);
pconf19=nfaTheorSim(ev19);
pconf20=nfaTheorSim(ev20);
pconf21=nfaTheorSim(ev21);
pconf22=nfaTheorSim(ev22);
pconf23=nfaTheorSim(ev23);
pconf24=nfaTheorSim(ev24);
pconf25=nfaTheorSim(ev25);

pconf=[pconf1; pconf2; pconf3; pconf4; pconf5; pconf6; pconf7; pconf8; pconf9; pconf10; pconf11; pconf12; pconf13; pconf14; pconf15; pconf16; pconf17; pconf18; pconf19; pconf20; pconf21; pconf22; pconf23; pconf24; pconf25]

x=[2 3 3.5 4];
x1=[2 2.8284 3.4641 4];
m=mean(pconf,1);

err1=pconf(:,1);
err2=pconf(:,2);
err3=pconf(:,3);
err4=pconf(:,4);

[H,P,C1]=ttest(err1,m(1),.05);
[H,P,C2]=ttest(err2,m(2),.05);
[H,P,C3]=ttest(err3,m(3),.05);
[H,P,C4]=ttest(err4,m(4),.05);

e = [C1(2)-C1(1) C2(2)-C2(1) C3(2)-C3(1) C4(2)-C4(1)]/2;

plot(x1,m,'ro-','LineWidth',3)
hold on
errorbar(x1,m,e, 'r')
publishfig

%title('Confusability as a Function of Theoretical Distance')
xlabel('Theoretical Distance')
ylabel('P(Intra-List Intrusion)')
