conf=[];
conf1=[];
conf2=[];
s=[];
s1=[];
s2=[];

i=1;
while i<=25
   mystr = ['events=ev' int2str(i) ';'];
   eval(mystr);
   [pconf, pconf1, pconf2, b, b1, b2]=nfaAbsLagSeven(events);
   conf=[conf; pconf];
   conf1=[conf1; pconf1];
   conf2=[conf2; pconf2];
   s=[s; b];
   s1=[s1; b1];
   s2=[s2; b2];
   i=i+1;
end

m=mean(conf,1)
m1=mean(conf1,1)
m2=mean(conf2,1)
  
mslope=mean(s,1)
m1slope=mean(s1,1)
m2slope=mean(s2,1)

err1=conf(:,1);
err2=conf(:,2);
err3=conf(:,3);
err4=conf(:,4);
[H,P,C1]=ttest(err1,m(1),.05);
[H,P,C2]=ttest(err2,m(2),.05);
[H,P,C3]=ttest(err3,m(3),.05);
[H,P,C4]=ttest(err4,m(4),.05);
err5=conf1(:,1);
err6=conf1(:,2);
err7=conf1(:,3);
err8=conf1(:,4);
[H,P,C5]=ttest(err5,m1(1),.05);
[H,P,C6]=ttest(err6,m1(2),.05);
[H,P,C7]=ttest(err7,m1(3),.05);
[H,P,C8]=ttest(err8,m1(4),.05);
err9=conf2(:,1);
err10=conf2(:,2);
err11=conf2(:,3);
err12=conf2(:,4);
[H,P,C9]=ttest(err9,m2(1),.05);
[H,P,C10]=ttest(err10,m2(2),.05);
[H,P,C11]=ttest(err11,m2(3),.05);
[H,P,C12]=ttest(err12,m2(4),.05);

e = [C1(2)-C1(1) C2(2)-C2(1) C3(2)-C3(1) C4(2)-C4(1)]/2;
e1 = [C5(2)-C5(1) C6(2)-C6(1) C7(2)-C7(1) C8(2)-C8(1)]/2;
e2 = [C9(2)-C9(1) C10(2)-C10(1) C11(2)-C11(1) C12(2)-C12(1)]/2;

x=[1 2 3 4];
subplot(1,3,1)
plot(x,m1,'ro-','LineWidth',3)
hold on
errorbar(x,m1,e1, 'r')
subplot(1,3,2)
plot(x,m2,'ro-','LineWidth',3)
hold on
errorbar(x,m2,e2, 'r')
subplot(1,3,3);
plot(x,m,'ro-','LineWidth',3)
hold on
errorbar(x,m,e, 'r')

publishfig


hold on

subplot(1,3,1);
xlabel('Absolute Lag')
ylabel('P(Intra-List Intrusion)')
title('First Half')
subplot(1,3,2);
title('Second Half')
subplot(1,3,3);
title('Combined')
 
[H,P,C]=ttest(s,0,.05);
[H,P1,C1]=ttest(s1,0,.05);
[H,P2,C2]=ttest(s2,0,.05);

P 
C
P1
C1
P2
C2
